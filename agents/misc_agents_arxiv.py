
import numpy as np, random
#import scipy.stats as stats # for truncated pareto

from collections import defaultdict, Counter
import time, math

# from flatland.dir_SMAs import SMAs
from flatland.SMAs.SMAs import DYAL, DistroQs

from flatland.envs.env_utils import get_move_from_to


# >class
class StrategyParams(object):
    # All the params of different strategies/agents in one place.

    # For the CompositeAgent
    
    # If none, means no budget (change strategy only if current
    # strategy returns None, ie fails).  for some strategies, in
    # particular random within a composite strategy, an initial
    # budget, of 1, is good..
    initial_budget =  None
    
    # When valid budget, multiply by the budget by this, every time
    # you get back to this.
    budget_multiple = 2 # 2 # 1 means fixed budget.

    # For any strategy requiring localization, should it use true
    # agent's local or the path integration of the agent.
    use_path_integration = True # False

    ########
    #
    # For the random and visit-count strategies.
    #
    # Setting to true improves random and visit-count strategies by a lot.
    #
    # do_biased_random = False # True
    always_remove_backward = True # do biased random

    ####
    #
    # Do reverse subsumption (the lower level over-riding the higher
    # level): in our context, it means when the agent is next to goal
    # (and within radius 1) it sees it, so it moves there. instead of
    # using the active strategy.

    # 
    do_override = 1 # 0 # True
    
    ########
    #
    # for the greedy strategy
    do_smell_greedy = True

    # For probabilistic memory/map ( prob-map ) strategy
    #
    perception_conf = 0.93 # 1.0 #0.93 # Initial confidence, from perception and localization
    #max_perception_conf = 1.0 #0.98 # Maximum achievable confidence
    
    item_symbols = ('b', 'f', 'e') # item or state symbols
    barrier_symbol, food_symbol, empty_symbol = item_symbols

    # For combining memory probabilities.  Each can be considered a
    # probability, and one way is to do a convex combination, but the
    # older the memory, the lower its weight (eg EMA). We could also
    # consider boosting confidence, if memories agree, so not just a
    # simple convex combination..
    
    # How fast/low should the weight of history/memory go down, with time?
    # 0 would mean completely ignore the  memory from previous days.
    day_beta = 0.1 # 0. # 0.1 # weights for exponential weighted (convex combination)
    # This one for within day cannot be higher than 0.5 (the most
    # recent should have highest weight).
    tick_beta = 0.4 # 0.0 # 0.4 # 0 would mean completely ignore the memory from previous ticks.
    ##
    via_perfs = True   # use performance to decide
    # when doing memory use via_perfs, whether a memory
    # beyond these many days should be ignored (when sampling barriers).
    too_old_for_use = None # (an integer if not None)

    # Keep track of two level probs.
    two_level_probs = 1 # False # True

    compress_multiple_ticks = 1 # True

    # planning
    num_plan_tries = 5 # If once thru goals is false.
    # go through goals in round robin manner?
    once_thru_goals =  False # True
    
    rough_plan_max_len = 50
    min_prob = 0.02
    food_min_prob = 0.01
    # Keep track of a current goal, and try planning towards it.
    support_current_goal = 0 # False
    
    #
    # when comparing memory against a simple prior.. (as a function of
    # day and item)
    #perf_beta = 0.10 # deprecated for now
    max_hist = 15 # beyond this many days, combine the memory.
    
    #
    # For quadratic distance, which memories to ignore
    min_qdist = 0.3

    # When using loss, which memories to ignore 
    max_loss_multiple = 1.5
    
    #
    # Pick the first plan that works.
    pick_first = True

    inday_q_cap = 10
    daily_q_cap = 5

    # to remove memories beyond this many days old.
    max_age = 5 # 1 # by day  ( max age )
    
    # Experimental only. Freeze the map of barriers after this day.
    last_update_day = None # Keep as None to continue updating..
    
    # Deprecated for now (batching can help if you want to randomize
    # order of updates, etc)
    # update_in_batch = 1 # True

    
####

def loc_str(loc): # assumes a pair
    if loc is None:
        return 'None'
    else:
        return '(%d, %d)' % (loc[0], loc[1])

#######



# >class
class CompositeAgent(object):
    """An agent that use a mix of strategies.
    """

    def __init__(
        self,
        env,
        strategies, # list of pure strategies
        seed=None
    ):
        assert strategies is not None and strategies != []
        
        self.sleeper = 0
        if seed is not None and seed != 0:
            random.seed(seed)
            np.random.seed(seed)

        # 0 is fwd in x ('west'), 1 is go up in y, 2 is back in x,
        # 3 is down in y (the absolute assumption)
        self.action_space = [0, 1, 2, 3]  # possible moves or actions
        self.hmode = env.render_mode  # human mode?

        self.shared_memory = SharedMemory(env)

        # The strategies list should not be empty.
        self.strategies = []
        i = 0
        for strategy in strategies:
            strat = self.select_strategy(strategy)
            self.strategies.append( strat )
            i += 1
            print('# %d.  strategy selected: %s' % (i, strat.get_name()) )

        print()
            
        self.num_strategies = len(self.strategies)
        # Initial time budgets, for each strategy.
        self.set_initial_time_budgets()
        # current_strategy is an integer index into the strategies array.
        self.current_strategy = 0
        
    ########### other fns

    # Should be called when environment changes.
    def set_env(self, env):
        self.shared_memory.set_env(env)
        #self.env = env
        # All strategies should have their environment set..
        #for strat in self.strategies:
        #    strat.set_env(env)
        
    # Allocate strategy object given choice of strategy (a
    # number). Environment (self.env) should have been set.
    # get_strategy
    def select_strategy(self, choice):
        shared = self.shared_memory
        # Choice of strategy
        if choice == 0:  # oracle or know-it-all
            return OracleStrategy(shared)
        elif choice == 1:  #
            return RandomStrategy(shared)
        elif choice == 2:
            return Greedy(shared)
        elif choice == 3:
            return VisitCounts(shared)
        elif choice == 4:
              return RememberPath(shared)
        elif choice == 5:
              return ProbMap(shared)
        else:
            return None

    # Do certain book keeping at the beginning of new day.
    # (prepare_for_next day!)
    def prepare_for_new_day(self, no_reward_yesterday=False):
        self.shared_memory.prepare_for_new_day(no_reward_yesterday)
        # Start with the strategies in order
        self.current_strategy = 0
        i = 0
        for strat in self.strategies:
            strat.prepare_for_new_day(no_reward_yesterday)
            # reset the time budgets at the beginning of each day.
            self.current_day_budgets[i] = self.beginning_of_day_budgets[i]
            self.time_spent_in[i] = 0
            i += 1
    
    # Do certain book keeping when a reward is found.
    def upon_obtained_reward(self, reward):
        assert reward > 0
        
        # Store current (estimated or true) location.

        # The (estimated) location needs to be updated, before target
        # is remembered.
        self.shared_memory.update_location()
        self.shared_memory.set_target_location()

        #if StrategyParams.use_path_integration:
        #    self.current_target = self.x, self.y
        #    #print(
        #    #    '\n# ?? reached reward, and reward loc set to:', self.ct_str(),
        #    #    ' status:', self.get_env_date_info() )
        #else:
        #    self.current_target = self.env.get_agent_location()

        for strat in self.strategies:
            strat.upon_obtained_reward() # self.current_target )


    # May return None: when all strategies are tried and all return
    # NONE! ( select_action )
    #
    # deprecated: obs is observations and reward is the reward if any,
    # from last action execution (or current time, before current
    # action execution).
    def get_action(self): # , obs, reward):
        # NOTE: we assume get_action is called exactly once for every
        # time point..

        # Do any update needed before action selection (and tell all
        # the strategies too).
        self.before_action_selection()

        self.shared_memory.increment_step() # += 1

        if StrategyParams.do_override:
            action = self.get_action_when_next_to_food()
            if action is not None:
                self.after_action_selection(action)
                return action

        # loops thru the strategies in order, until one yields a
        # non-None action (or all have been tried).
        i = 0
        while i < self.num_strategies:
            strat = self.strategies[self.current_strategy]
            if 0:
                print('\n# step:', self.step, ' loc:', self.x, self.y, ' current strat: ', strat.get_name(),
                  ' current_spent: ', self.time_spent_in[self.current_strategy],
                  ' curren goal: ', self.current_target)
            action = strat.get_action( ) # obs, reward
            if action is not None:
                self.time_spent_in[self.current_strategy] += 1
                # self.decrement_budget(self.current_strategy)

            if action is None or self.is_times_up(self.current_strategy):
                # Move to next strategy
                self.change_strategy()
                print( '# strategy changed to:',
                       self.current_strategy, ' spent_time:',
                       self.time_spent_in[self.current_strategy],
                       ' budget on strategy:',
                       self.current_day_budgets[self.current_strategy] )
                
            if action is not None:
                break
            i += 1

        self.after_action_selection(action)
        return action

    ###

    # this is used for subsumption. If agent is next to food, then go
    # to food! (Return the action)..
    def get_action_when_next_to_food(self):
        return self.shared_memory.env.get_action_when_next_to_food()

    ####

    def before_action_selection(self):
        # If path integration is used, current location is already
        # updated via update_my_location() (upon selection of action
        # in the last time point).

        self.shared_memory.update_location()
            
        # loc = (self.x, self.y)
        for strat in self.strategies:
            strat.before_action_selection()
    
    # Upon (ie after) selection of action: let all strategies know
    # (what action was taken, where...)
    def after_action_selection(self, action):
        self.shared_memory.update_last_location()
        self.shared_memory.update_last_action(action)
        for strat in self.strategies:
            strat.after_action_selection( action)
        
    ###
    
    # NOTE: num strategies should have been set.
    def set_initial_time_budgets(self):
        self.beginning_of_day_budgets = [
            StrategyParams.initial_budget for _ in range(self.num_strategies)]
        # Identical to beginning of the day, but during the day it be raised!
        self.current_day_budgets = [
            StrategyParams.initial_budget for _ in range(self.num_strategies)]
        # This is set at current_day_budgets, whenever used strategy
        # switches to this strategy, and goes down when the strategy
        # is used!
        
        self.time_spent_in = [0 for _ in range(self.num_strategies)]

        i = 0
        for b in self.current_day_budgets:
            self.strategies[i].set_time_budget(b)
            i += 1

    def set_budgets(self, budgets):
        if budgets is None or budgets == []:
            return # Nothing to do
        i = 0
        for budget in budgets:
            if budget <= 0: # don't change the default
                i += 1
                continue
            self.beginning_of_day_budgets[i] = budget
            self.current_day_budgets[i] = budget
            self.time_spent_in[i] = 0
            self.strategies[i].set_time_budget(budget)
            i += 1
        
    # Is it times up for this strategy?
    def is_times_up(self, strat):
        budget = self.current_day_budgets[strat]
        if budget is not None:
            return self.time_spent_in[strat] >= budget
        else:
            return False

    # update the time budget of the strategy (for next time this
    # strategy is used). Reset time spent in this strategy.
    def update_budget(self, strat, budget=None):
        self.time_spent_in[strat] = 0 # reset
        # Set to spcific value?
        if budget is not None:
            self.current_day_budgets[strat] = budget
            return
        
        budget = self.current_day_budgets[strat]
        if budget is None: # no time budget?
            return
        self.current_day_budgets[strat] *= StrategyParams.budget_multiple
        self.strategies[strat].set_time_budget(self.current_day_budgets[strat])
        
    ####

    # Currently, just moves to next strategy with a wrap-around.
    # (cycles thru strategies in order), but before that, may set a
    # new time budget for current strategy.
    # 
    # Returns the array index of the new strategy.
    def change_strategy(self):
        # Update the time budget for the next time we get back to this
        # strategy.
        if 1:
            print( '# today:', self.shared_memory.get_day() ,
                   'strategy is changing from:',
                   self.current_strategy, ' spent_time:',
                   self.time_spent_in[self.current_strategy],
                   ' current budget on this strategy:',
                   self.current_day_budgets[self.current_strategy] )
 
        self.update_budget(self.current_strategy)
        
        self.current_strategy += 1
        # Cycle thru the strategies, in order
        self.current_strategy %= self.num_strategies

        # Invoke preparation of the new strategy (if any).
        self.strategies[self.current_strategy].upon_switch_to_this()

        return self.current_strategy

########

# >class
class SharedMemory:
    def __init__(self, env):
        self.env = env
        # My current location (possibly an estimate
        self.x, self.y = env.home # should often be 0, 0
        self.current_target = None
        # steps (actions taken) within a day
        self.step = 0
        # steps (actions taken) within a day
        self.day = 1

        self.last_action = None
        # for biased random, etc
        self.last_location = None

    def get_shared_info_str(self):
        goal = None
        if self.current_target:
            x, y = self.current_target
            goal = '(%d, %d)' % (x, y)
        x, y = self.get_location()
        loc = '(%d, %d)' % (x, y)
        x, y = self.get_true_location()
        true_loc = '(%d, %d)' % (x, y)
        return ('\n#  day:%d step:%d, agent_loc:%s (trueLoc:%s), goal:%s last_action:%s' %
                (self.day, self.step, loc, true_loc, goal, self.last_action  ))

    def get_last_location(self):
        return self.last_location

    def update_last_location(self):
        self.last_location = self.get_location()
        
    def update_last_action(self, action):
        self.last_action = action

    def prepare_for_new_day(self, no_reward_yesterday=False):
        self.x, self.y = self.env.home
        self.step = 0
        self.day += 1
        self.last_action = None
        #print('# HERE7, current target:', self.current_target)

    def set_env(self, env):
        self.env = env

    def get_today(self):
        return self.day
        
    def get_day(self):
        return self.day
        
    # Get agent's current estimate of its own location 
    def get_location(self):
        return (self.x, self.y)

    def get_true_location(self):
        return self.env.get_agent_location()

    # For path integration (assumes action is legal or None).  return
    # the old and new location (coordinates). NOTE: if path
    # integration is not selected (is false), then the new estimate is
    # over-written in the next step by the actual location.
    def update_location(self):
        if not StrategyParams.use_path_integration:
            self.set_location_to_true_location()
            return
        
        if self.last_action is None:
            assert self.step == 0
            return
        
        action = self.last_action
        old_loc = self.get_location() # (self.x, self.y)
        x, y = old_loc
        if action == 0:  # fwd move (horizontal)
            x += 1
        if action == 2:  # back move
            x += -1
        if action == 1:  # (vertical) up move (or down depending on view)
            y += 1
        if action == 3:  # (vertical) down move (or up depending on view)
            y += -1
        # set to the estimated location
        self.set_location(x, y)
      
    def increment_step(self):
        self.step += 1

    def get_legal_actions(self):
        return self.env.get_legal_actions()
    
    def set_location_to_true_location(self):
        self.x, self.y = self.env.get_agent_location()
    
    def set_location(self, x, y):
        self.x, self.y = x, y
        
    def set_target_location(self, loc=None):
        self.current_target = loc
        if loc is None:
            self.current_target = self.get_location()


########################
########## Pure Strategies

# >class
class PureStrategy():
    # This is the generic/abstract pure strategy class, and we'll put
    # some common fields functions here. Every concrete pure strategy
    # should implement it.
    #
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory

    def before_action_selection(self):
        pass

    # Invoke preparation when this strategy is switched to (if any).
    def upon_switch_to_this(self):
        pass
    
    def get_time_budget(self):
        return self.time_budget
    
    def set_time_budget(self, budget):
        self.time_budget = budget
    
    def get_today(self):
        return self.shared_memory.day

    # time step, or time tick
    def get_step(self):
        return self.shared_memory.step
    
    def after_action_selection(self, action):
        pass

    def get_agent_info_str(self):
        return self.shared_memory.get_shared_info_str()

    def upon_obtained_reward(self):
        # self.set_current_target( target )
        pass
    
    # NOTE: we assume get_action is called exactly once for every
    # time point..
    def get_action(self):
        # All child classes should have this implemented..
        assert False
        
    def get_legal_actions(self):
        return self.shared_memory.get_legal_actions()

    def prepare_for_new_day(self, no_reward_yesterday=False):
        pass
        #self.x, self.y = self.env.home

    # Get agent's estimated location.
    def get_my_location(self):
        return self.shared_memory.get_location()

    def get_true_location(self):
        return self.shared_memory.get_true_location()

    # Assumes there is food (won't insert food).
    def get_closest_true_food_loc(self):
        x, y = self.get_true_location()
        foods = self.shared_memory.env.get_closest_food(
            x, y, insert_if_not=False)
        assert len(foods) > 0, '# no food closest to %s.. food count:%d\n\ngrid:\n%s' % (
            loc_str((x, y)),
            self.shared_memory.env.get_food_count(),
            self.shared_memory.env.barriers_to_text() )  

        return foods[0]
    
    # For biased and visit count random walk .. 
    def remove_backward_action(self, actions):
        la = len(actions)
        if la <= 1:
            return actions
        # We could also use the last action (if that's recorded).
        # (perhaps if the last action wasn't used.. or perception said
        #  it wasnot successful, then here this is an alternative...)
        my_last = self.shared_memory.get_last_location()
        x1, y1 = self.get_my_location( )
        kept = [] # kept moves
        # Should the backward move be removed or just down-weighed?
        for move in actions:
            # see where it would end up..
            x2, y2 = self.convert_action_to_loc(x1, y1, move)
            if (x2, y2) != my_last:
                kept.append(move)
        assert kept != []
        return kept

    # For implementing a biased random walk and visit-count, we need a
    # bit of memory (of the last location).  (could be a queue too)
    #def update_location_info(self, old_loc):
    #    self.last_location = old_loc
    #    #self.x, self.y = new_loc
    #    return

    # What is the next location given the 'move' action is executed on
    # current x1, y1 (used in a few places).
    @classmethod
    def convert_action_to_loc(cls, x1, y1, move):
        if move == 0:
            return x1 + 1, y1
        if move == 1:
            return x1, y1 + 1
        if move == 2:
            return x1 - 1, y1
        if move == 3:
            return x1, y1 - 1
        # Should not reach here.
        assert False


####

# >class
class RandomStrategy(PureStrategy):
    
    def __init__(self, env):
        super().__init__(env)

    def get_name(self):
        return 'RandomStrat'

    def get_action(self):
        actions = self.get_legal_actions()
        if actions == [] or actions is None:
            return None
        if StrategyParams.always_remove_backward:
            # actions = self.make_biased_random_moves(actions)
            actions = self.remove_backward_action(actions)

        np.random.shuffle(actions)
        #for a in actions:
        #    assert self.shared_memory.env.is_action_legal(a), \
        #   ' in rand.. action was: %s' % str(action)

        return actions[0]

#######

# >class
class Greedy(PureStrategy):
    def __init__(self, shared_memory):
        super().__init__(shared_memory)

    def get_name(self):
        return 'Greedy'
    
    # in greedy
    def get_action(self):
        greedy = self.get_greedy_moves(
            use_my_location=not StrategyParams.do_smell_greedy,
            use_my_target=not StrategyParams.do_smell_greedy,
        )
        if greedy == []:
            return None
        np.random.shuffle(greedy)
        return greedy[0]
 
    def get_greedy_moves(self, use_my_location=1, use_my_target=1):
        # Note: use_my_location is set independent of whether we are
        # doing/using path integration.
        # (to determine the greedy direction)

        #print('# HERE use my target:', use_my_target)
        if use_my_location:
            x1, y1 = self.get_my_location()
        else:
            # If using 'smell', etc. (get the true location to get the
            # true direction)
            x1, y1 = self.shared_memory.get_true_location()

        # similarly for target/goal location
        if use_my_target:
            if self.shared_memory.current_target is not None:
                x2, y2 = self.shared_memory.current_target
                #print('HERE8 my own current target:', self.shared_memory.current_target)
            else:
                return []
        else:  # current_target is None, which may imply no more food.
            foods = self.shared_memory.env.get_closest_food(insert_if_not=False)
            if foods == []:
                return []
            #print("HERE2, got true food loc:", foods[0])
            x2, y2 = foods[0]

        # Set all to false. Figure the greedy directions.
        fwdx, fwdy, bkx, bky = 0, 0, 0, 0
        fwdx = x2 > x1  # move fwd in x (true if this is a greedy step)
        fwdy = y2 > y1  # move fwd in y  (true if this is a greedy step)
        bkx = x2 < x1  # move backward in x
        bky = y2 < y1  # move backward in y
        
        # NOTE: if target==current agent location, or there are barriers,
        # greedy fails.
        
        # get all possible greedy+legal actions
        return self.greedy_moves_given_info(fwdx, fwdy, bkx, bky)

    def greedy_moves_given_info(self, fwdx, fwdy, bkx, bky):
        # check which actions in the greedy direction, if any, are
        # legal (no barrier, etc..)
        greedy = [] # greedy moves

        # Get the true legal greedy moves.
        legals = self.get_legal_actions()
        if fwdx and 0 in legals:
            greedy.append(0)
        if fwdy and 1 in legals:
            greedy.append(1)
        if bkx and 2 in legals:
            greedy.append(2)
        if bky and 3 in legals:
            greedy.append(3)

        return greedy

#######

####

# >class
class VisitCounts(PureStrategy):
    """The visit-counts strategy favors adjacent cells that have low
    visit counts (the unvisited adjacent cells). Leads to better
    exploration than random.

    """
    
    def __init__(self, env):
        super().__init__(env)
        self.visit_counts = Counter()

    def get_name(self):
        return 'VisitCounts'

    def prepare_for_new_day(self, no_reward_yesterday=False):
        if not no_reward_yesterday:
            self.clear_visited_counts()  # reset visit history.
        else:
            # (if last day was no_reward_yesterday, keep exploring..)
            print('\n# new day: but not clearing the visit counts!\n')
        super().prepare_for_new_day(no_reward_yesterday)

    # in least visited (assuming end of day)
    def upon_obtained_reward(self):
        today = self.get_today()
        print('# today:', today, '\tnum_visit_counts:\t', len(self.visit_counts))

        
    # reset_visit... 
    def clear_visited_counts(self):
        self.visit_counts = Counter()

    def before_action_selection(self):
        self.update_visited( self.get_my_location() )
        super().before_action_selection()
            
    # Should we make this probabilistic with SMA? what would it mean?
    def update_visited(self, loc):
        self.visit_counts[loc] += 1

    def after_action_selection(self, action):
        #print("HERE9: myloc:", self.get_my_location(), ' oldloc:', old_loc,
        #      ' integ:',  StrategyParams.use_path_integration)
        super().after_action_selection(action)

    # in visit-counts
    def get_action(self):
        x1, y1 = self.get_my_location()
        actions = self.get_legal_actions()
        if actions == [] or actions is None:
            return None

        if StrategyParams.always_remove_backward:
            actions = self.remove_backward_action(actions)
         
        if len(actions) > 1:
            np.random.shuffle(actions)
            actions = self.rank_by_visited_count(actions)
        
        return actions[0]

    def rank_by_visited_count(self, actions):
        x1, y1 = self.get_my_location()
        to_sort = []
        for move in actions:
            # see where it would end up..
            x2, y2 = self.convert_action_to_loc(x1, y1, move)
            count = self.visit_counts.get((x2, y2), 0)
            to_sort.append((count, move, (x2, y2) ))
        # sort by count, lowest first (prioritize unvisited states/cells)
        to_sort.sort(key=lambda x: x[0])
        return [x[1] for x in to_sort]

####

# Path Memory or PathMem, etc..
#
# >class
class RememberPath(PureStrategy):
    """Remember yesterday's path (ie the last action taken at each location). """    
    def __init__(self, env):
        super().__init__(env)
        self.loc_to_action = {}
        self.path = {}
        self.is_visited = {}

    def get_name(self):
        return 'RememberPath'

    def after_action_selection(self, action):
        loc = self.get_my_location()
        self.remember_action(loc, action)
        self.is_visited[loc] = True
        super().after_action_selection(action)

    def upon_obtained_reward(self):
        self.set_path_info( self.loc_to_action )
        super().upon_obtained_reward()
        
    def remember_action(self, loc, action):
        # override anything that was there..
        self.loc_to_action[ loc ] = action
        
    def prepare_for_new_day(self, no_reward_yesterday=False):
        self.loc_to_action = {}
        self.is_visited = {}
        super().prepare_for_new_day(no_reward_yesterday)

    # Agent calls this before next day.
    def set_path_info(self, loc_to_action):
        # remember yesterday's path (basically, transfer loc_to_action
        # to path (which is a map too)
        self.path = {}
        for loc, action in loc_to_action.items():
            self.path[ loc ] = action

    def get_remembered_action(self, x1, y1):
        action = self.path.get( (x1, y1), None )
        return action

    # in RememberPath
    def get_action(self):
        x1, y1 = self.get_my_location()
        action = None
        # If already was here we assume we used the recommended
        # action (from yesterday) and so don't use it again..
        # (the path is not working..)
        if (x1, y1) not in self.is_visited:
            action = self.get_remembered_action(x1, y1)

        # remember visit-count, and used to fall back to
        # least-visited, or some other strategy, if the last action
        # didnt work. (maybe relevant to remember_path strategy only?)
        
        legals = self.get_legal_actions()
        
        if (  # there recommended action is not legal?
            action is None or action not in legals
        ):
            return None        
        return action

####

# >class
class ProbMap(PureStrategy):
    """Remember barrier/food/empty locations, use them as as map for
    planning and replanning.

    """
    def __init__(self, shared_memory):
        super().__init__(shared_memory)
        #self.loc_to_memory = {}
        self.plan = {}

        # For the 3 object types, Empty, Barrier, Food.
        # (currently not used, just informational)
        # self.prior_sma = SMAs.DYAL()
        self.prior_sma = DYAL()

        self.current_goal = None
        self.current_budget = 0
        
        # predictions of each mem type, changes *within* the day.
        self.todays_preds = {} # at the start of the day, uses daily preds.

        # The daily (over days, not internal to day!).
        # this changes at the end of the day (more stable)
        self.daily_preds = {}
        
        # daily losses of the mem types
        self.daily_losses = {} # mem_type->loss
        # two level loss
        self.todays_losses = {}
        
        # memory before newest memory for a location.
        # self.loc_to_memory_before = {}

        # from location to list of memories
        self.loc_to_mems_list = defaultdict(list)
        self.foods_today = []
        self.to_update = []
        #
        # (currently deprecated) combined memory up to yesterday
        self.combined_yesterday_memory = {}

        # For information
        self.day_to_num_plans = Counter() # number of plannings in a day
        self.day_to_max_goals = Counter()
        
    def get_name(self):
        return 'ProbMap'

    def after_action_selection(self, action):
        if 0:
            today = self.get_today()
            step = self.get_step()
            print('# today:%d, %d' % (today, step),
                  ' num total updates (to do after):', len(self.to_update))
            
        super().after_action_selection(action)

    
    def upon_obtained_reward(self):

        if StrategyParams.compress_multiple_ticks:
            self.compress_multiple_mems()

        self.loss_update_count = {}
        today = self.get_today()
        if len(self.day_to_num_plans) > 0:
            vals =  list(self.day_to_num_plans.values())
            print('# today:', today, '\tnum_plannings:\t', self.day_to_num_plans[today],
                  '\tmedian so far:\t', np.median(vals), '\tmax:\t', np.max(vals)  )
            
        print('# today:', today, ' num_max_goals:\t', self.day_to_max_goals[today] )
        if 1: # print food locs with their probs
            fprobs= []
            foods = self.get_food_probs()
            foods.sort(key=lambda x: -x[1]) # descending in prob
            for loc, prob, _ in foods:
                fprobs.append( (loc_str(loc), '%.2f' % prob) )
            print('# today:', today, ' food_probs:\t', len(fprobs), fprobs)


        
        assert StrategyParams.two_level_probs
        if StrategyParams.two_level_probs:
            self.fold_in_into_daily(pit=0)
                # self.aggregate_within_day_then_daily(pit=1)
        else:
            for _, perf, item in self.to_update:
                perf.update(item)
        

        if 1:
            print('# end_of_day %d, num_mem_types:%d\n' % (today, len(self.daily_preds)))
            for mem_type, pred in self.daily_preds.items():
                print('# end_of_day %d' % self.get_today(),
                      ' daily distro for:', mem_type,
                      distro_str( pred.get_distro()))

             

                
        super().upon_obtained_reward()

    
    # Compress today's memories (memories from previous days are
    # assumed compressed already). Compression means: for the same
    # location, but different times/ticks, we could get multiple
    # memories (observing the same or different objects).
    def compress_multiple_mems(self):
        # For the same day, get one memory per item-type.
        today = self.get_today()
        sp = StrategyParams
        new_map = defaultdict(list)
        processed = set()
        num_mem1, num_mem2 = 0, 0
        for loc, mem_lists in self.loc_to_mems_list.items():
            num_mem1 += len(mem_lists)
            distros = defaultdict(list)
            for mem in mem_lists:
                mem_day = mem.get_day()
                if mem_day < today:
                    # We assume already compressed
                    new_map[loc].append(mem)
                    # Make sure previous days, we have at most one
                    # memory for this location..
                    #assert (loc, mem_day) not in processed, '%d %d' % (mem_day, today)
                    #processed.add( (loc, mem_day) )
                    continue
                item = mem.get_remembered_item()
                distros[item].append( (mem.get_distro(), None) )

            # for now, for each item type, does insert one (and ignore
            # how many of each type observed)..
            for item, dis in distros.items():
                # distro = self.aggregate_predictions( dis )
                # max_sym, max_prob = max_prob_entry(distro)
                compressed_mem = ProbMemory(
                    item == sp.barrier_symbol,
                    item == sp.food_symbol, today, 1)
                new_map[loc].append( compressed_mem )
            num_mem2 += len(new_map[loc])

        self.loc_to_mems_list =  new_map
        print("\n# end of day:%d, num_mems before, after compression of today's:\t%d\t%d" % (
            today, num_mem1, num_mem2))
        
    def fold_in_into_daily(self, pit=0):
        if pit:
            today = self.get_today()
            print('# today:%d' % today, ' num total updates:', len(self.to_update))

        # 'fold in' changes of today into the daily
        for mem_type, pred in self.todays_preds.items():
            predictor = self.daily_preds.get(mem_type, None)
            if predictor is None:
                predictor = allocate_fractional_Q(StrategyParams.daily_q_cap)
                self.daily_preds[mem_type] = predictor
            predictor.update( pred.get_distro() )
            if 0:
                print('# end_of_day %d' % self.get_today(),
                      ' in-day distro for:', mem_type,
                      distro_str( pred.get_distro()),
                      ' daily-distro updated to:',
                      distro_str( predictor.get_distro()))
            
        # Now make new copies for next day
        for mem_type, pred in self.daily_preds.items():
            self.todays_preds[mem_type] = pred.copy(5)

        # Do the same for losses. (update the daily)
        for mem_type, todays in self.todays_losses.items():
            loss = self.daily_losses.get(mem_type, None)
            if loss is None:
                loss = Scalar_EMA(0.03) #
                self.daily_losses[mem_type] = loss
            loss.update( todays.get_val() )
        
        # Now make new copies for next day
        for mem_type, loss in self.daily_losses.items():
            l2 = Scalar_EMA(0.1, val=loss.get_val(), rate=0.1)
            self.todays_losses[mem_type] =  l2


    # First aggregate the observations with each other (separately for
    # each mem type) (this is very batch
    def aggregate_within_day_then_daily_old(self, pit=0):
        within_day = {} # from mem_type to todays_predictor
        todays_losses = defaultdict(list)
        if pit:
            today = self.get_today()
            print('# today:%d' % today, ' num total updates:', len(self.to_update))
            
        for mem_type, _, item in self.to_update:
            predictor = within_day.get(mem_type, None)
            was_none = False
            if predictor is None:
                was_none = True
                # predictor = SMAs.DYAL() # Dyal is not quick enough for food...
                # (we could use DYAL for 'empty' and 'barrier' ? 
                predictor = allocate_fractional_Q()
                within_day[mem_type] = predictor
            # predictor.update(item)
            
            # Now measure loss
            daily_pred = self.todays_preds.get( mem_type, None)
            if daily_pred is not None:
                todays_losses[mem_type].append(self.compute_loss(daily_pred, item))                
            elif not was_none:
                todays_losses[mem_type].append(self.compute_loss(predictor, item))

            
            predictor.distro_update( {item: 1.0} ) # For fractional Qs
            #print('# ', mem_type, 'for', item, ' pred distro after update:', distro_str(predictor.get_distro()) )

            #if item == 'f':
            #    print('\n# day:', self.get_today(), ' a mem was updated for food:', mem_type)
            #if mem_type[0] == 'f':
            #    print('\n day:', self.get_today(),' a mem had food as its item:', mem_type)
                
        for mem_type, pred in within_day.items():
            daily_pred = self.todays_preds.get( mem_type, None)
            if daily_pred is None:
                daily_pred = allocate_fractional_Q()
                self.todays_preds[mem_type] = daily_pred
            if 1 or pit:
                print('# HERE76 today:%d' % today, ' for mem:', mem_type,
                      ' update_count:', pred.get_update_count(),
                      ' distro:', distro_str(pred.get_distro()) )
            # Combine this current distro (from today's) with
            # the moving average, for the current memory type.
            daily_pred.distro_update( pred.get_distro() )
            if pit:
                print('     combined distro:', distro_str(daily_pred.get_distro()) )
            loss_list = todays_losses[mem_type]
            if loss_list == []:
                #print('# HERE12 loss list was empty..')
                continue
            daily_loss = self.todays_losses.get( mem_type, None)
            if daily_loss is None:
                daily_loss = Scalar_EMA(0.03)#
                self.daily_losses[mem_type] = daily_loss
            daily_loss.update( np.mean(  loss_list   ) )
            self.loss_update_count[mem_type] =  len(  loss_list   ) 
            
    # Start with log loss.
    def compute_loss(self, predictor, item):
        prob = max(predictor.get_prob(item), 0.01)
        return -math.log(prob)
        
            
    # Mix/aggregate today's and yesterday's (combined) memory
    # (in the model for keeping one memory)
    def update_combined_memory(self):
        # The probabilities should be combined but also degraded too..
        combined = {}
        for loc, mem in self.loc_to_memory.items():
            last_memory = self.combined_yesterday_memory.get(loc, None)
            if last_memory is  None:
                combined[loc] = mem
            else:
                mem.combine_probs(last_memory, clear_beta=False)
                combined[loc] = mem
                
        for loc, mem in self.combined_yesterday_memory.items():
            if loc in self.loc_to_memory:
                continue # already processed.
            combined[loc] = mem
            
        self.combined_yesterday_memory = combined
        self.loc_to_memory = {}

    def prepare_for_new_day(self, no_reward_yesterday=False):
        self.foods_today = self.get_food_locs()
            # today=self.get_today(), # today ..
            #min_prob=StrategyParams.min_prob )

        self.plan={}
        self.current_goal = None        
        if 1:
            self.show_predictions()
            
        self.curtail_memories() # prune memories!

        super().prepare_for_new_day(no_reward_yesterday)

    def upon_switch_to_this(self):
        # See if food locations have expanded/changed, etc.
        self.foods_today = self.get_food_locs()
        self.plan={}
        self.current_goal = None

    # drop memories that are too old.
    def curtail_memories(self):
        today = self.get_today()
        for loc, mems in self.loc_to_mems_list.items():
            new_mems = []
            for mem in mems:
                if today - mem.get_day() > StrategyParams.max_age:
                    continue
                new_mems.append(mem)
            self.loc_to_mems_list[loc] = new_mems

    # Look around you and updates probabilistic map.
    def before_action_selection(self):
        super().before_action_selection()
        if StrategyParams.last_update_day is not None:
            if self.get_today() > StrategyParams.last_update_day:
                # no more updating (for experimental purposes)
                return
        self.update_prob_memory( )

    def update_prob_memory(self ):
        x, y = self.get_my_location()
        # Get the true location, to get objects
        tx, ty = self.shared_memory.get_true_location()

        # Update for the four cells around you.
        self.update_probs_for_loc(x+1, y, tx+1, ty)
        self.update_probs_for_loc(x-1, y, tx-1, ty)
        self.update_probs_for_loc(x, y+1, tx, ty+1)
        self.update_probs_for_loc(x, y-1, tx, ty-1)
        # skip the location you are on.. it will be seen once you move
        # to an adjacent cell..
        # self.update_probs_for_loc(x, y, tx, ty)

    def update_probs_for_loc( self, x, y, tx, ty, update_prior=True ):
        #
        # tx and ty are the true agent viewing location, to use to see
        # whether there is barrier at that location.
        #
        is_barrier = self.shared_memory.env.is_barrier_at_loc(tx, ty)
        is_food = self.shared_memory.env.is_food_at_loc(tx, ty)
        self.update_predictions(
            (x, y), is_barrier, is_food)
        
        if update_prior:
            #if is_food:
            #    print('HEREf prior is updated_for_food!', self.get_today() )
            self.prior_update( is_barrier, is_food )
            
        self.loc_memory_update( (x, y), is_barrier, is_food )

    # Update the predictions, as a function of memory type
    # (what it votes for, and its age)
    def update_predictions(
            self, loc, is_barrier, is_food):
        mems = self.loc_to_mems_list.get(loc, [] )
        if mems == []: # nothing to do.
            return
        
        today = self.get_today()

        sp = StrategyParams
        item = sp.empty_symbol
        if is_barrier:
            item = sp.barrier_symbol
        elif is_food:
            item = sp.food_symbol
            
        for memo in mems:
            age = today - memo.get_day() 
            self.update_mem_type_distro(
                item, memo.get_remembered_item(), age)

    ###
    
    # NOTE: this was already updated to computed conditional distros,
    # given memory type! (but i decided to abondon it).
    def update_prediction_performance_old(
            self, loc, is_barrier, is_food):
        memo = self.loc_to_memory.get(loc, None)
        if memo is None:
            return # nothing to do

        today = self.get_today()
        diff = today - memo.get_day() 

        sp = StrategyParams
        item = sp.empty_symbol
        if is_barrier:
            item = sp.barrier_symbol
        elif is_food:
            item = sp.food_symbol
            
        # p1 = memo.get_prob(item)
        prediction = memo.get_remembered_item()
        
        #p2 = self.prior_sma.get_prob(item)
        
        # self.update_comparison_old(item, p1, p2, diff)
        self.update_mem_type_distro(item, prediction, diff)

        mems = self.loc_to_mems_list.get(loc, [] )
        #if 1 and memo2 is not None:
        for mem2 in mems:
            diff2 = today - mem2.get_day() 
            #p3 = memo2.get_prob(item)
            #self.update_comparison_old(item, p3, p2, diff2)
            self.update_mem_type_distro(
                item, memo2.get_remembered_item(), diff2)

    ###

    # Update for the given memory type (type is designated by age,
    # what the memory votes for, etc).
    def update_mem_type_distro(self, true_item, remembered, age):
        sp = StrategyParams
        if age > sp.max_hist: # no more than k days?
            age = sp.max_hist

        mem_type = (remembered, age)

        # Given what's predicted by memory, is it true?
        predictor = self.todays_preds.get( mem_type, None)
        if predictor is None:
            #predictor = SMAs.DYAL()
            if age == 0:
                predictor = allocate_fractional_Q(StrategyParams.inday_q_cap)
            else:
                earlier = age - 1
                predictor = self.daily_preds.get( (remembered, earlier) , None)
                # this assertion can fail in blank grids!
                #assert predictor is not None or remembered=='f', \
                #    ' remed:%s age:%d' % (remembered, age)
                if predictor is None:
                    predictor = allocate_fractional_Q(StrategyParams.inday_q_cap)
                    predictor.update( { remembered:1.0 } )
                else:
                    # Use the distro from mem-type a day before, for
                    # initialization.
                    predictor = predictor.copy(5)
                
            self.todays_preds[ mem_type ] = predictor

        #if sp.update_in_batch:
        #    self.to_update.append( ( mem_type, predictor, true_item)) # schedule the update
        #else:
            
        loss = self.todays_losses.get(mem_type, None)
        if loss is None:
            if loss is None:
                loss = Scalar_EMA(0.03) #
                self.todays_losses[mem_type] = loss
        loss.update(self.compute_loss(predictor, true_item))
        predictor.update( {true_item:1.0}, False ) # update the distro

        
        """
        if predicted == true_item:
            perf.update('1')
        else:
            perf.update('0')
        """
        

        

    ###
    
    # p1 is the memory.. p2 is prediction of the prior (here, we
    # assume the memory is probabilistic.. but this approach wasn't
    # useful.. we need to compute the accuracy of memory.. given
    # memory says something, how often is it correct.. )
    def update_comparison_old(self, item, p1, p2, diff):
        sp = StrategyParams
        if diff > sp.max_hist: # no more than 5 days
            diff = sp.max_hist
        perf = self.todays_preds.get((item, diff), None)
        if perf is None:
            perf = SMAs.DYAL()
        #perf *= (1 - sp.perf_beta) # down weigh
        #if p1 > p2:
        #    perf += sp.perf_beta
        if p1 > p2:
            perf.update('1')
        else:
            perf.update('0')
        self.todays_preds[(item, diff)] = perf

    ####

    # Show different distributions, eg where the food is, and for each
    # mem-type, its predictions.
    def show_predictions(self):
        print('\n# food memory distros, etc  on day=%d num_possible_foods:%d :\n' %
              (self.get_today(), len(self.foods_today)  ) )
        #for loc in self.foods_today:
        print('# food locs: ', [loc_str(loc) for loc in self.foods_today])
        # ' p:%.2f' % trip[1], '  had_supporting_confidence:%d' % trip[2] )
        print('\n')
        print('# prior: ', distro_str( self.prior_sma.get_distro()  ))
        print('\n# place memory distros, etc  on day=%d :\n' %
              (self.get_today() ) )

        max_len = 0
        for loc, mems in self.loc_to_mems_list.items() :
            if len(mems) > max_len:
                max_len = max_len
                max_loc = loc
        print('\n# max_len:%d max_loc:%s\n' % (max_len, loc_str(max_loc)))
        
        pairs = list(self.todays_preds.items())
        for pair, predictor in pairs:
            #print( pair, ' num_updates:', predictor.get_update_count(),
            #       ' distro:', distro_str(predictor.get_distro() ))
            print( pair,   ' distro:', distro_str(predictor.get_distro() ))
            loss_ema = self.daily_losses.get(pair, None)
            if loss_ema:
                print('#     the mean loss is: %.3f' % loss_ema.get_value())
            else:
                print('#     no mean-loss')

        pairs = list(self.daily_losses.items())
        pairs.sort( key=lambda x: x[1].get_value())
        print('\n# sorted losses for day ', self.get_today())
        for mem_type, loss_ema in pairs:
            print('# mem_type:', mem_type, ' loss: %.2f' % loss_ema.get_val(),
                  '  update_count:', loss_ema.get_update_count())
            print('#    num_loss_updates:', self.loss_update_count.get(mem_type, 0))

        print()
        
    def show_perfs_old(self, thrsh=0.7):
        print('\n# memory PERFORMANCES on day=%d above t=%.2f:\n' %
              (self.get_today(), thrsh) )
        pairs = list(self.todays_preds.items())
        pairs = [
            (x[0], x[1].get_prob('1'), x[1]) for x in pairs]
        pairs.sort(key=lambda x: -x[1])
        i = 0
        for pair, val, perf in pairs:
            i += 1
            print('# %d. %s, perf:%.2f %d' % (
                i, pair, val, perf.get_edge_update_count('1') ) )
            if val <= thrsh:
                break
        
    def prior_update(self, is_barrier, is_food ):
        sp = StrategyParams
        if is_food:
            self.prior_sma.update(sp.food_symbol)
        elif is_barrier:
            self.prior_sma.update(sp.barrier_symbol)
        else:
            self.prior_sma.update(sp.empty_symbol)

    # Update the probabilistic map/memory for this location.
    def loc_memory_update(self, loc, is_barrier, is_food ):
        # Combine the last memory, with what is observed now.
        #last_memory = self.loc_to_memory.get(loc, None)

        # 1st: save (preserve) it to 'one before last'
        #if last_memory is not None: 
        # self.loc_to_memory_before[loc] = last_memory
        #self.loc_to_mems_list[loc].append( last_memory )

        ## Now combine
        #self.loc_to_memory[loc] = self.combine_memory_and_latest(
        #    last_memory, is_barrier, is_food)

        mem = self.combine_memory_and_latest(None, is_barrier, is_food)
        #self.loc_to_memory[loc] = mem
        self.loc_to_mems_list[loc].append( mem )


    # Combine the current observation with the last memory
    def combine_memory_and_latest(
        self, last_memory, is_barrier, is_food):
        mem_today = ProbMemory(is_barrier, is_food, self.shared_memory.day,
                               self.shared_memory.step)
        return mem_today.combine_probs(last_memory)

    def get_food_locs(  self):
        locs = []
        for loc, mems in self.loc_to_mems_list.items():
            for memo in mems:
                o = memo.get_remembered_item()
                if o != StrategyParams.food_symbol:
                    continue
                else:
                    locs.append(loc)
                    break
        return locs
           
    def get_item_prob_pairs(
            self, item, today=None, min_prob=0):
        loc_to_p = {}
        loc_to_p_no_conf = {}
        for loc, mems in self.loc_to_mems_list.items():
            found = 0
            for memo in mems:
                o = memo.get_remembered_item()
                if o != item:
                    continue
                else:
                    found = 1
                    break
            if found: # a loc/memory of food is found.
                print('# day=', today, ',  found a loc with mem-food:',
                      loc_str(loc), ' num mems:', len(mems))
                prob_map, has_conf = self.extract_probs_from_memories(
                    mems, today=today, use_distro=1, item=item, pit=1 )
                if prob_map is None:
                    continue
                print('# prob-map is:', distro_str(prob_map))
                p = prob_map.get(item, 0)
                if p > 0:
                    p2 = loc_to_p.get(loc, 0)
                    if p2 < p:
                        if has_conf:
                            loc_to_p[loc] = p
                        else:
                            loc_to_p_no_conf[loc] = p
        # Shouldn't we just return all?
        locs1 = list([(x[0], x[1], True) for x in loc_to_p.items() ])
        locs2 = list([(x[0], x[1], False) for x in loc_to_p_no_conf.items() ])
        locs = locs1 + locs2
        return locs
    
    def find_item(self, item, prob_map, min_prob, x=None, y=None, processed=None ):
        focus = []
        for loc, memo in prob_map.items():
            if processed is not None:
                if loc in processed:
                    continue
            prediction = memo.get_remembered_item()
            if prediction != item:
                continue
            
            if p1 <= min_prob:
                continue
            if loc == (x, y): # skip that location
                #print('#  skipping food location since already there..!  loc:',
                #      x, y)
                continue
            focus.append( (loc, p1) )
        return focus

    def find_item_old(self, item, prob_map, min_prob, x=None, y=None, processed=None ):
        focus = []
        for loc, memo in prob_map.items():
            if processed is not None:
                if loc in processed:
                    continue
            p1 = memo.get_prob(item)
            if p1 <= min_prob:
                continue
            if loc == (x, y): # skip that location
                #print('#  skipping food location since already there..!  loc:',
                #      x, y)
                continue
            focus.append( (loc, p1) )
        return focus

    # in ProbMap strategy
    def get_action(self):
        x1, y1 = self.get_my_location()
        #print('\n#HERE39 plan size: %d\n#   %s' %
        #      (len(self.plan), self.get_agent_info_str() ))

        action = self.plan.get( (x1, y1), None )
        legals = self.get_legal_actions()
        
        if action is None or action not in legals:
            self.plan, goal = self.make_plan()
            if goal is not None:
                self.shared_memory.set_target_location(goal)

        action = self.plan.get( (x1, y1), None )
        if action is None or action not in legals:
            return None

        if StrategyParams.support_current_goal:
            self.current_budget -= 1
            if self.current_budget <= 0:
                self.current_goal = None
            
        return action

    # Just sort (possibly filter) and return the pairs.
    def weighted_order_foods(self, pairs, min_prob=None):
        if min_prob is not None and min_prob > 0:
            pairs = list(filter(lambda x: x[1] >= min_prob, pairs))
        lp = len(pairs)
        if lp <= 1: # shouldn't happen?
            return pairs
        pairs.sort(key=lambda x: -x[1])
        return pairs
        

    def pick_a_food_loc(self, pairs, i=None, min_prob=None):

        if min_prob is not None and min_prob > 0:
            pairs = list(filter(lambda x: x[1] >= min_prob, pairs))
        
        lp = len(pairs)
        if lp == 0: # shouldn't happen?
            return None, None, None
        if lp == 1:
            return pairs[0] # [0]

        # First few times, pick by highest prob ..
        if i is not None and i <= 1:
            pairs.sort(key=lambda x: -x[1])
        else: # otherwise random shuffle, and pick by prob at random.
            np.random.shuffle(pairs)
        #print('\n# HERE13: food probs were:' , ['%.3f'% x[1] for x in pairs] )
        # Sample probabilisitically. todo: we can make it a function
        # of nearness and likelihood.
        p1 = np.random.uniform(0, 1)
        sump = 0
        for loc, p2, had_conf in pairs:
            sump += p2
            if p1 <= sump:
                #print('# HERE13: picked food with prob: %.3f' % p1)
                return loc, p2, had_conf
        
        #print('# HERE13: picked food with prob: %.3f' % pairs[0][1])
        return pairs[0]

    def make_plan(self):
        today = self.get_today()
        
        x, y = self.get_my_location()
        sp = StrategyParams
        #print('# HERE45 Making a new PLAN!')
        #self.show_prob_memory()
        #foods = self.get_item_prob_pairs(
        #    sp.food_symbol, min_prob=sp.min_prob, x=x, y=y)

        goal_prob = 0
        if self.current_goal is not None:
            if self.current_goal == (x, y):
                self.current_goal = None
                
        if self.current_goal is not None:
            res = self.get_food_prob( self.current_goal )
            goal_prob =  0
            if res is not None:
                goal_prob =  res[1]
            if goal_prob < sp.food_min_prob:
                goal_prob = 0
                self.current_goal = None
        
        if self.current_goal is None:
            if self.foods_today == []:
                print('# Num foods today was 0..')
                return {}, None
            else:
                print('# num initial foods today: %d' % len(self.foods_today))

            foods = self.get_food_probs()
            
            print('# num foods after extracting current probs: %d' % len(foods))
            if foods == []:
                return {}, None
            
            self.day_to_max_goals[today] = max( len(foods), self.day_to_max_goals[today] )
        
        # temporary, for testing
        # foods = [((14, 14), .3, 1),  ((13, 14), .3, 1), ((14, 13), .3, 1), ((15, 15), .3, 1)]

        num_tries = sp.num_plan_tries # planning tries
        if sp.once_thru_goals and self.current_goal is None:
            foods = self.weighted_order_foods( foods, sp.food_min_prob )
            num_tries = len(foods)
        
        i, selected = 0, {}
        goal = None
        while i < num_tries:
            if self.current_goal is None:
                if sp.once_thru_goals: # go thru goals one by one.
                    goal, prob, _ = foods[i]
                else:
                    goal, prob, _ = self.pick_a_food_loc(foods, i, sp.food_min_prob)
            else:
                goal = self.current_goal
                prob = goal_prob
                
            #goal = 14, 14,         # temporary, for testing
            if goal is not None:
                print('#HERE55 picked food goal:', goal, ' with_prob:%.3f:' % prob)
            else:
                break
            
            #barriers = self.sample_prob_map(sp.barrier_symbol, sp.min_prob, goal)
            barriers = self.sample_barriers_map(sp.min_prob, goal)

            plan = self.make_path_from_map_memory(
                goal=goal, barriers_memory=barriers)
            i += 1
            if plan != {} and sp.pick_first: # done
                selected = plan
                break
                #return plan, goal
                
            if plan != {} and (selected == {} or len(plan) < len(selected)):
                selected = plan

        # if current goal is not none, do not reset current budget!
        if sp.support_current_goal and self.current_goal is None and selected != {}:
            self.current_goal = goal
            self.current_budget = 2 * len(plan)

        if selected != {}:
            self.day_to_num_plans[today] += 1
        return selected, goal

    def show_prob_memory(self):
        print('\n# agent_info %s' % self.get_agent_info_str())
        #print('# prior:' % self.
        self.prior_sma.print_distro()
        print('\n# map memory: ')
        for loc, memo in self.loc_to_memory.items():
            print('# loc:%s, %s' % (
                loc_str(loc), memo.get_str() ) )

    # Get a list of food probs (for planning, etc).
    def get_food_probs(self):
        foods = []
        for food_loc in self.foods_today:
            foods.extend( [ self.get_food_prob(food_loc)] )
        foods = list( filter(lambda x: x is not None, foods) )
        return foods

            
    # Get food prob. for the given loc. (it may change during the
    # day)..  
    def get_food_prob(self, food_loc):
        mems = self.loc_to_mems_list.get(food_loc, [])
        
        prob, has_conf = self.extract_food_memories(
            mems, food_loc, pit=0)
        #if prob_map is None:
        #    return []
        #p = prob_map.get(StrategyParams.food_symbol, 0)
        #if p < StrategyParams.min_prob:
        #    return []
        if prob > 0:
            return  (food_loc, prob, has_conf)
        return None

    def extract_food_memories( self, mems,  loc=None, pit=0 ):
        sp = StrategyParams
        num1, num2 = 0, 0
        best_no_conf = False
        best_p = 0
        #if today is None:
        today = self.get_today() # shared_memory.day
        for memo in mems:
            memo_day = memo.get_day()
            diff = today - memo_day
            remed_item = memo.get_remembered_item()
            mem_type = (remed_item, diff)
            predictor = self.todays_preds.get( mem_type ,  None )
            distro = {}
            if predictor is not None:
                distro = predictor.get_distro()
            if predictor is None or distro == {}:
                if remed_item != sp.food_symbol:
                    continue
                print('\n# day:', self.get_today(), ', step:', self.get_step(),
                      ', for food_loc mem_type is:', mem_type,
                      " memory-day and  time tick:", memo_day, memo.get_step()   )
                num1 += 1
                best_no_conf = True
                #distros.append( (distro, None, diff, mem_type ) )
                continue
            else:
                if pit:
                    print('# mem_type:%s, Today: %d, before: %s' % (
                        mem_type, today, distro_str(distro)) )
                p = distro.get(sp.food_symbol, 0)
                num2 += 1

                print('\n# day:', self.get_today(), ', step:', self.get_step(),
                      ', for food_loc mem_type is:', mem_type, ' p:', '%.3f' % p,
                      " memory-day and  time tick:", memo_day, memo.get_step()   )

                if p > best_p:
                    best_p = p

        print('# num-mems with food at', loc_str(loc) , ', num1 and num2:', num1 , num2)
        
        if best_p > 0:
            return best_p, True
        if best_no_conf:
            return 1.0, False
        return 0, None

    def extract_max_barrier_prob_from_memories( self, mems, today=None, do_prior=False, pit=0 ):
        if today is None:
            today = self.get_today() # shared_memory.day
        barrier = StrategyParams.barrier_symbol
        maxp = 0
        processed_types = set()
        for memo in mems:
            memo_day = memo.get_day()
            diff = today - memo_day
            remed_item = memo.get_remembered_item()
            mem_type = (remed_item, diff)
            # there can be multiple from the same day (probably need
            # to be aggregated first, before folding into daily..)
            if mem_type in processed_types:
                continue
            processed_types.add( mem_type )
            predictor = self.todays_preds.get( mem_type ,  None )
            if predictor is None:
                continue
            distro = predictor.get_distro()    
            if distro == {}:
                continue
            p = distro.get(barrier, None)
            if p is None:
                continue
            if p > maxp:
                maxp = p
        #print('# --> maxp:%.2f' % maxp)
        return maxp
        
            
    # today may be sent in, as next day (when preparing for new day).
    def extract_probs_from_memories( self, mems, use_loss=1, today=None, pit=0 ):
        if today is None:
            today = self.get_today() # shared_memory.day
        distros = []
        processed_types = set()
        for memo in mems:
            memo_day = memo.get_day()
            diff = today - memo_day
            remed_item = memo.get_remembered_item()
            mem_type = (remed_item, diff)
            # there can be multiple from the same day (probably need
            # to be aggregated first, before folding into daily..)
            if mem_type in processed_types:
                continue
            processed_types.add( mem_type )
            predictor = self.todays_preds.get( mem_type ,  None )
            distro = {}
            if predictor is not None:
                distro = predictor.get_distro()
                
            if predictor is None or distro == {}:
                distro = {remed_item:1.0}
                distros.append( (distro, None, diff, mem_type ) )
                continue
            else:
                if pit:
                    print('# mem_type:%s, Today: %d, before: %s' % (
                        mem_type, today, distro_str(distro)) )
                    
                distro = self.populate_rest(distro)
                if use_loss:
                    w = self.todays_losses.get( mem_type, None )
                    if w is not None:
                        w = w.get_value()
                        if pit:
                            print('# HERE88, w or loss is: %.3f' % w)
                else:
                    # for now, skip this.
                    assert False
                    if 0:
                        w = self.distance_to_prior(distro)
                        if pit:
                            print('# Today: %d, dist:%.2f, after: %s' % (
                                today, w, distro_str(distro)))
                distros.append( ( distro, w, diff, mem_type ) )

        return self.pick_and_aggregate(distros, use_loss=use_loss)

    # Assign a weight to missing items in distro (as long as the
    # original distro sent in is a semi distribution)
    def populate_rest(self, distro):
        l = len(distro)
        sp = StrategyParams
        if l >= len(sp.item_symbols):
            return distro
        sump = sum(distro.values())

        assert l > 0 and sump < 1.03, 'l:%d and sump:%.4f, distro:%s' % (
            l, sump, distro_str(distro) )
        
        if sump >= 1.0:
            return distro
        
        rem1 = 1.0 - sump # remainder prob. mass
        accounted = 0.0
        for item in sp.item_symbols:
            if item in distro:
                accounted += self.prior_sma.get_prob(item)

        rem2 = 1.0 - accounted
        if rem2 <= 0.0001:
            return distro
        distro2 = {}
        for item in sp.item_symbols:
            if item in distro:
                distro2[item] = distro[item]
                continue
            distro2[item] = rem1 * self.prior_sma.get_prob(item) / rem2
        return distro2


    # experimental/not used.
    # could be KL, or quadratic, or use the binomial tail
    def distance_to_prior(self, distro):
        # Do quadratic loss for starters..
        s = 0
        for item, p1 in distro.items():
            p2 = self.prior_sma.get_prob(item)
            d = p1 - p2
            s += d * d
        # remaining items
        for item, p in distro.items():
            if item in distro:
                continue
            s += p * p
        return math.sqrt(s)

        
    def pick_and_aggregate(self, distros, use_loss=True):
        l = len(distros)
        if l <= 1:
            if l == 0:
                return None, None
            else:
                if distros[0][1] is None:
                    return distros[0][0], False
                else:
                    return distros[0][0], True

        # When there are some distributions (memories) that have
        # confidence/distance (from prior), then drop those that
        # don't.
        #assert len(distros) > 0
        distros2 = list( filter(lambda x: x[1] is not None, distros ) )
        l = len(distros2)
        if l == 0:
            assert len(distros) > 0
            return self.aggregate_no_confidences_case(distros), False
        if l <= 1:
            return distros2[0][0], True
        
        # Keep the ones that are sufficiently far from the (moving)
        # prior and also from each other (non-redundant memories, for
        # the purposes of prediction)..
        if use_loss:
            distros3 = distros2 # no change
        else: # the larger the distance, the better
            distros3 = list( filter(lambda x: x[1] > StrategyParams.min_qdist, distros2 ))
        if distros3 == []: # all are short distance from prior..
            assert not use_loss # could not have been set..
            distros2.sort(key=lambda x: -x[2]) # sort by increasing diff/age
            return distros2[0][0], True # pick the first/most recent
        else:
            if use_loss:
                distros3.sort(key=lambda x: x[1]) # sort by increasing loss
                if 0: # pick one best (lowest loss)..
                    # print('# HERE66 top 3 losses: ', [ '%.3f %s' % (x[1], x[3]) for x in distros3[:3] ] )
                    return distros3[0][0], True
                else: # aggregate
                    best = distros3[0][1] # lowest loss
                    picked, multiple = [], StrategyParams.max_loss_multiple
                    for tup in distros3:
                        if tup[1] <= multiple * best:
                            picked.append(tup)
                    return self.aggregate_predictions(picked), True
            if 1:
                distros3.sort(key=lambda x: -x[1]) # sort by decreasing dist
                # highest first..
                print('# HERE66 top 3 losses: ', [ '%.3f %s' % (x[1], x[3]) for x in distros3[:3] ] )
                return distros3[0][0], True
            else:
                # We haven't done this when using loss...
                assert not use_loss
                distros3.sort(key=lambda x: -x[2]) # sort by diff/age
                # could remove those too close to one another..?
                return self.aggregate_predictions(distros3), True
            
    # distros is a list of 3 tuples, the first component being the
    # distro..
    def aggregate_predictions(self, distros):
        agg = Counter() # the aggregation
        sump = 0.0
        for sd in distros:
            sd = sd[0]
            for o, p in sd.items():
                agg[o] += p
                sump += p
        for o, p in agg.items():
            agg[o] /= sump
        return agg
            
            
        
    # Initially, we won't have confidences for the different
    # memory types, so we assume they put all their mass on their
    # single item vote. We assume here, all distros are of this type..
    def aggregate_no_confidences_case(self, distros):
        # This should not happen much (only initially)..  A simple way
        # is just to random permute and pick one..
        random.shuffle(distros)
        return distros[0][0]    

    # use_probs ( for barriers , etc )
    def sample_barriers_map(self,  min_prob=0, goal=None):
        sp = StrategyParams
        item = sp.barrier_symbol
        today = self.get_today()
        sampled = {} # a key-value map
        j = 0
        n_barrs = 0
        for loc, mems in self.loc_to_mems_list.items():
            if goal is not None and loc == goal:
                continue # Don't put a barrier there
            j += 1
            max_sym = None
            if 1:
                prob_map, p = None, 0
                if 0:
                    p = self.extract_max_barrier_prob_from_memories(mems, today=today, pit=0)
                else:
                    prob_map, has_conf = self.extract_probs_from_memories(mems, pit=0)
                    
                if prob_map is not None:
                    #print( '#   HERE123, prob map is:', distro_str(prob_map) )
                    max_sym, max_prob = max_prob_entry(prob_map)
                    # When barrier wins, set its prob to 0.9..?
                    #if max_sym == sp.barrier_symbol:
                    #    prob_map[sp.barrier_symbol] = max(.69, max_prob) 
            else:
                prob_map = { sp.barrier_symbol: 0.3 }

            #if max_sym is not None:
            #    #print('HERE144  max symbol:', max_sym)
            #    if max_sym == sp.barrier_symbol:
            #        sampled[loc] = True
            #    continue
            
            #p = self.get_prob_to_use(item, memo)
            if prob_map is not None:
                p = prob_map.get(sp.barrier_symbol, 0)
            #else:
            #p = self.prior_sma.get_prob(sp.barrier_symbol)
            #   p = 0
                
            if p > min_prob and np.random.uniform(0, 1) < p:
                sampled[loc] = True
                n_barrs += 1
                
        ## print('\n# HERE123DONE! num in loop:%d, num bariers: %d\n' % (j, n_barrs))
        
        return sampled

    # currently deprecated
    def sample_prob_map_old(self, item, min_prob=0, goal=None):
        sampled = {} # a key-value map
        for loc, memo in self.loc_to_memory.items():
            if goal is not None and loc == goal:
                continue # Don't put a barrier there
            p = self.get_prob_to_use(item, memo)
            if p > min_prob and np.random.uniform(0, 1) < p:
                sampled[loc] = True
                #sampled.append(loc)
    
        sp = StrategyParams
        if sp.two_level_probs: # use yesterday's combined memory too.
            for loc, memo in self.combined_yesterday_memory.items():
                if loc == goal or loc in self.loc_to_memory:
                    continue
                p = self.get_prob_to_use(item, memo)
                if p > min_prob and np.random.uniform(0, 1) < p:
                    sampled[loc] = True
        return sampled

    def make_path_from_map_memory(self, goal, barriers_memory):
        plan = self.plan_a_path( goal, barriers_memory )
        # return self.map_calculated_path.get((x1, y1), None)
        # The plan (or path) is a map from location to action
        if plan is None:  
            return {}
        return plan

    def plan_a_path(self, target, barriers_memory):
        x1, y1 = self.get_my_location()
        # NOTE for prob. memory, there could be multiple locations
        # with nonzero target probs (even when there is exactly one
        # true location), and one can still plan.. perhaps the caller
        # of this functions need to make sure such a distinct target is picked.
        if target == (x1, y1):  # Already at target? (the target may
            # not have food, could have been an exploratory target, or
            # bad/noisy localization)
            print(
                '\n# 88888 I think I am already at target!! (may switch to another strategy)\n'
            )
            return {}

        if 1: # a fix when bool_vs_list
            # make sure barrier and agent location do not overlap
            # self.remove_if_barrier_overlaps(goal=target)
            # dim = self.env.get_dimension()
            b = self.get_time_budget()
            if b is None:
                len_limit = StrategyParams.rough_plan_max_len
            else:
                len_limit = 3 * b # plan limit

            print('\n# path planning.. len_limit:%d (time_budget:%s)' % (len_limit, str(b) ))
            path = utils_find_a_path(
                (x1, y1), target, barriers_memory,
                len_limit=len_limit, mems=self.loc_to_mems_list )
            if path is None or len(path) == 0:
                print('# path planning was NOT sucessful..')
            else:
                print('# path planning was sucessful.. len: %d' % len(path))
            return path

"""
    @classmethod
    def my_find_a_path(cls, target, x1, y1, barrier_memory, max_multiple=5, len_limit=10):
        x2, y2 = target  # goal
        # print('# in path planning, target: ', target)
        # Put start (x1, y1) in queue
        #
        # Repeat: Take 1st node out from queue. for each neib, if not
        # yet visited, insert in the queue with an estimate to goal,
        # sort queue by estimate to goal.
        queue, visited, last = [((x1, y1), 0, None, None)], set(), None
        parent = {}  # Allows us to construct the path
        itr = 0
        # If a node is too far from target, as a multiple of straight
        # grid path length to goal, don't add it to queue.  (This way,
        # we do not need to know the dimensions of the grid.. )

        # max_dist = max(max_multiple * (abs(x1 - x2) + abs(y1 - y2)), 2 * dim)
        max_dist = len_limit
        max_dist_found = 0
        while queue != []:
            itr += 1
            node = queue.pop()  # the last is shortest estimate
            node_xy = node[0]
            dist_to_src = node[1]
            visited.add(node_xy)
            #parent[node_xy] = node[2]
            parent[node_xy] = node[3]
            neibs = cls.get_neibs_via_memory(barrier_memory, node_xy)
            for n in neibs:
                if n == (x2, y2):  # done: reached the target
                    last = node_xy
                    break
                # Dont add children (nodes) that are too far ..
                # ( it's an indication that you are going out of
                #   the environment )
                dist_to_goal = abs(n[0] - x2) + abs(n[1] - y2)
                if dist_to_goal > max_dist:
                    max_dist_found = max(dist_to_goal, max_dist_found)
                    continue
                if n not in visited:
                    visited.add(n)
                    #queue.append((n, dist, node_xy))
                    queue.append((n, dist_to_src+1, dist_to_goal, node_xy))

            if last:
                break
            # Shortest estimated distance last
            queue.sort(key=lambda x: -(x[1]+x[2]) )  # lowest last
            # print('# queue is now:', queue)
            if itr % 100 == 99:
                print('# .. continuing the loop:', itr, len(queue))
                print('# .. target is:', x2, y2, '  max dist is:', max_dist)
                if queue != []:
                    print('# .. last queue entry:', queue[-1])
                    print('# .. 1st queue entry:', queue[0])

        # End of while loop
        # print('\n# itrs in planning:', itr)
        if last is None:  # no path was found
            print('\n# did NOT find  path.. (from ', x1, y1, 'to', x2, y2, ')')
            print('# max dist found (the node may been skipped): ', max_dist_found )
            return None

        # String/costruct the path, meaning the action to take at each
        # location, on the path from start to finish
        #
        # last is the node right before goal
        node_xy, planned_move, plan = last, {}, 0
        while node_xy is not None:
            planned_move[node_xy] = get_move_from_to(node_xy, (x2, y2))
            x2, y2 = node_xy
            node_xy = parent.get(node_xy, None)
            plan += 1
        print('# found a path of length: ', plan, 'from', x1, y1, 'to', target)
        return planned_move

"""

####
#

#
# A single memory, represented with uncertainty (distribution).
# 
# >class
class ProbMemory:
    """Prob. distro for one location. Although, depending on the
    technique, we may not use the probabilities. """
    def __init__(self, is_barrier, is_food, today, time_now):
        self.day = today
        self.time_tick = time_now
        self.item_to_prob = {}
        sp = StrategyParams
        # Assign initial probability (confidence).
        confidence = sp.perception_conf
        self.remembered_item = None
        if is_barrier:
            self.item_to_prob[sp.barrier_symbol] = confidence
            self.remembered_item = sp.barrier_symbol
        elif is_food:
            self.item_to_prob[sp.food_symbol] = confidence
            self.remembered_item = sp.food_symbol
        else:
            self.item_to_prob[sp.empty_symbol] = confidence
            self.remembered_item = sp.empty_symbol
            
        # Now populate probability for the remaining symbols
        rem_mass = (1.0 - confidence) / ( len(sp.item_symbols) - 1 )
        for symb in sp.item_symbols:
            if symb not in self.item_to_prob:
                self.item_to_prob[symb] = rem_mass

    
    def get_day(self):
        return self.day
    
    def get_step(self):
        return self.time_tick
    
    def get_str(self):
        s = ''
        for item, prob in self.item_to_prob.items():
            s += ' %s:%.2f' % (item, prob)
        s += ' day:%d time:%d' % (self.day, self.time_tick)
        return s

    # Combine current memory/observation with last memory (for the
    # same location).
    def combine_probs(self, last_memory, clear_beta=True):
        if last_memory is None:
            return self
        
        sp = StrategyParams
        a = 0 # weight of last memory
        day_beta = sp.day_beta
        if sp.two_level_probs and clear_beta:
            day_beta = 0 # don't combine yesterday's with today's
            
        if last_memory.day < self.day:
            if day_beta > 0: # If 0, ignore last memory
                a = math.pow( day_beta, self.day - last_memory.day )
        else:
            # If tick_beta is 0, ignore last (within day) memory..
            if sp.tick_beta > 0:
                a = math.pow( sp.tick_beta, self.time_tick - last_memory.time_tick )

        if a <= 0: # no modification needed.
            return self
        
        combined = {}
        for item, prob1 in self.item_to_prob.items():
            prob2 = last_memory.item_to_prob[item]
            combined[item] = (1-a) * prob1 + a * prob2

        self.item_to_prob = combined
        return self


    def get_distro(self):
        return self.item_to_prob.copy()
    
    def get_prob(self, item):
        return self.item_to_prob.get(item, 0)

    # The item that was observed (with highest prob) when the memory
    # was recorded.
    def get_remembered_item(self):
        return self.remembered_item

    
    """
        remed_item = None
        maxp = 0
        for o, p in self.item_to_prob.items():
            if p > maxp:
                maxp = p
                remed_item = o
        return remed_item
    """

#####

# >class
class OracleStrategy(PureStrategy):
    """The Oracle agent knows or has access to the exact map, and true
    agent location.

    """
    def __init__(self, shared_memory):
        super().__init__(shared_memory)
        self.plan = {}
        
    def get_name(self):
        return 'Oracle'

    def after_action_selection(self, action):
        super().after_action_selection(action)

    def upon_obtained_reward(self):
        # reset plan for next day (so it's computed
        # again).
        self.plan = {}
        super().upon_obtained_reward()

    # Look around you and updates probabilistic map.
    def before_action_selection(self):
        super().before_action_selection()
        
    # in Oracle
    def get_action(self):
        x1, y1 = self.get_true_location()
        action = self.plan.get( (x1, y1), None )
        legals = self.get_legal_actions()

        assert action is None or action in legals, 'action was: %s, true_loc:%s' % (
            action, loc_str((x1, y1)))
        # It's possible the map is not made yet, or agent went of
        # course, due to motion noise.
        if action is None:
            self.plan = self.make_oracle_plan()
        
        action = self.plan.get( (x1, y1), None )
        if action is None or action not in legals:
            return None
        #print('# action was:', action)
        return action

    def make_oracle_plan(self):
        x, y = self.get_true_location()
        
        food_loc = self.get_closest_true_food_loc()
        assert food_loc is not None

        barriers = self.shared_memory.env.gather_barrier_coords()
        
        max_dist = 10 * self.shared_memory.env.get_size()
        plan = utils_find_a_path( (x, y), food_loc, barriers, len_limit=max_dist)
        assert plan != {}
        print('\n# plan: ', plan)
        return plan
 
######

# barriers memory is in the form of a map.
def utils_find_a_path(
        src, target, barriers_memory, len_limit=50, mems=None, pit=0):
    x1, y1 = src
    x2, y2 = target  # goal
    # print('# in path planning, target: ', target)
    # Put start (x1, y1) in queue
    #
    # Repeat: Take 1st node out from queue. for each neib, if not
    # yet visited, insert in the queue with an estimate to goal,
    # sort queue by estimate to goal.
    queue, visited, last = [((x1, y1), 0, None, None)], set(), None
    parent = {}  # Allows us to construct the path
    itr = 0
    # If a node is too far from target, as a multiple of straight
    # grid path length to goal, don't add it to queue.  (This way,
    # we do not need to know the dimensions of the grid.. )

    # max_dist = max(max_multiple * (abs(x1 - x2) + abs(y1 - y2)), 2 * dim)
    max_dist = len_limit
    max_dist_found = 0 # before reaching the goal, some nodes are this far?? ...
    queue_limit = 4 * len_limit
    
    #print('\n# HERE6 num barriers: %d, src:%s' % (
    #        len(barriers_memory), loc_str(src)) )
    #for loc in barriers_memory.keys():
    #    print('# Barrier at: %s' % loc_str(loc))
        
    while queue != []:
        if len(queue) > queue_limit:
            break
        itr += 1
        node = queue.pop()  # the last is shortest estimate
        node_xy = node[0]
        dist_to_src = node[1]
        visited.add(node_xy)
        #parent[node_xy] = node[2]
        parent[node_xy] = node[3]
        neibs = get_neibs_via_memory(barriers_memory, node_xy)
        for n in neibs:
            if n == (x2, y2):  # done: reached the target
                last = node_xy
                break
            if mems is not None:
                if n not in mems:
                    continue
            # Dont add children (nodes) that are too far ..
            # ( it's an indication that you are going out of
            #   the environment )
            dist_to_goal = abs(n[0] - x2) + abs(n[1] - y2)
            if dist_to_goal > max_dist:
                max_dist_found = max(dist_to_goal, max_dist_found)
                continue
            if n not in visited:
                visited.add(n)
                if (itr > 1000):
                    print('# adding loc:%s to queue during search.. qSize:%d est_dist_to_goal:%d  max_dist:%d' % (
                        loc_str(n), len(queue), dist_to_goal, max_dist) )
                #queue.append((n, dist, node_xy))
                queue.append((n, dist_to_src+1, dist_to_goal, node_xy))

        if last:
            break
        # Shortest estimated distance last
        queue.sort(key=lambda x: -(x[1]+x[2]) )  # lowest last
        # print('# queue is now:', queue)
        if 1 and itr % 100 == 99:
            print('# .. continuing the loop:', itr, len(queue))
            print('# .. target is:', x2, y2, '  max dist is:', max_dist)
            if queue != []:
                print('# .. last queue entry:', queue[-1])
                print('# .. 1st queue entry:', queue[0])

    # End of while loop
    # print('\n# itrs in planning:', itr)
    if last is None and pit:  # no path was found
        print('\n# did NOT find full  path.. (from ', x1, y1, 'to', x2, y2, ')')
        print('# max dist found (the node may been skipped): ', max_dist_found )
        return None

    # String/costruct the path, meaning the action to take at each
    # location, on the path from start to finish
    #
    # last is the node right before goal
    node_xy, planned_move, plan = last, {}, 0
    child = {} # child map for printing the plan
    while node_xy is not None:
        planned_move[node_xy] = get_move_from_to(node_xy, (x2, y2))
        if node_xy is not None:
            child[node_xy] = (x2, y2)
        x2, y2 = node_xy
        node_xy = parent.get(node_xy, None)
        plan += 1


    if pit or 0:
        print('\n\n# found a path of length: ', plan, 'from',
              x1, y1, 'to', loc_str(target))
        node = src
        while node != target:
            print('# node:%s action:%s next:%s' % (
                loc_str(node), planned_move[node], loc_str(child[node])))
            node = child[node]
        print('\n')
        
    return planned_move


def get_neibs_via_memory(barriers_memory, node_xy):
    x, y = node_xy
    # blocks or walls or barriers!
    walls = get_barriers(barriers_memory, node_xy)

    neibs = []
    # Do we assume the agent knows the size of the environment?
    if (x + 1, y) not in walls:  # and x+1 < env.size:
        neibs.append((x + 1, y))
    if (x, y + 1) not in walls:
        neibs.append((x, y + 1))
    if (x - 1, y) not in walls:  # and x-1 >= 0:
        neibs.append((x - 1, y))
    if (x, y - 1) not in walls:  # and y-1 >= 0:
        neibs.append((x, y - 1))
    return neibs


def get_barriers(barrier_memory, node_xy):
    x, y = node_xy
    bars = set([])
    if barrier_memory.get((x+1, y), None):
        bars.add((x+1, y))
    if barrier_memory.get((x-1, y), None):
        bars.add((x-1, y))
    if barrier_memory.get((x, y+1), None):
        bars.add((x, y+1))
    if barrier_memory.get((x, y-1), None):
        bars.add((x, y-1))
    return bars
 

def distro_str(distro, sort=1):
    s = ''
    pairs = list(distro.items())
    if sort:
        pairs.sort(key=lambda x: -x[1])
    for e, p in pairs:
        s += ' %s:%.3f' % (e, p)
    return s

# arg max
def max_prob_entry(distro):
    max_s, max_v = None, None
    for s, value in distro.items():
        if max_v is None or value > max_v:
            max_v = value
            max_s = s
    return max_s, max_v

def allocate_fractional_Q(cap=StrategyParams.inday_q_cap):
    # queue of distribution.
    # return SMAs.DistroQs( q_capacity=cap )
    return DistroQs( q_capacity=cap )
    


# >class
class Scalar_EMA:
    def __init__(self, min_rate, val=0, rate=1.0):
        self.min_rate = min_rate
        self.rate = rate # initial rate
        self.value  = val
        self.updates = 0
        
    def update(self, obs):
        self.updates += 1
        self.value *= (1.0 - self.rate)
        self.value += self.rate * obs
        self.rate = 1.0 / (1.0 / self.rate + 1.0)
        self.rate = max(self.rate, self.min_rate)

    def get_value(self):
        return self.value

    def get_val(self):
        return self.value
    
    def set_val(self, val):
        self.value = val

    def set_rate(self, rate):
        self.rate = rate

    def get_update_count(self):
        return self.updates


####




