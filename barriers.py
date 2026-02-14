import argparse
import os
import numpy as np
import sys

# Example run.
#
# Need to set the path, for example
# export PYTHONPATH=$PYTHONPATH:/Users/...../dir_flatland_forage/git_flatland
#
#
# python3 barriers.py --size 15 --prop 0.2 --sts 1,2
#
# One can inject noise into the agent motion
# (-noise 0.03) and use path integration (-integ 1) to measure
# (estimate) agent location (average results over 150 initial
# environments, 50 days each, and -noh means do it in 'batch', ie
# non-human display, mode).
#
# python3  barriers.py --size 15  -be --prop 0.3 -chr 0.0  -noise 0.03 -noh  -integ 1 \
#  -ms 5500 -integ 1  -sts 3,4 -gprb 0.50   -fbr 0 -dsg 1  -days 50 -outer 150 | tail
#
# use -atd when oracle agent is picked ( -atype 0 ) (also for smell-greedy, if you want it to use
# on day 1)
# 
import time

# can be commented out for now.
# from PIL import Image  # we are using it for saving

from flatland.envs.base_grid_env import BaseGridEnv

import flatland.make_envs as make_envs

from envs.barriers_env import BarriersEnv

# from envs import EnvUtils
from envs.env_utils import get_env_with_a_path

from agents.misc_agents_arxiv import CompositeAgent, StrategyParams

# from agents.misc_agents import MiscAgents


description = """Code for barrier experiments: python3 barriers.py -h  """

#

# timer!
def get_mins_from( from_time ):
    delta = time.time() - from_time
    return delta / 60.0

def get_hrs_from( from_time ):
    delta = time.time() - from_time
    return delta / 3600.0

#
def parse_args():
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('filename')           # positional argument
    # parser.add_argument('-c', '--count')      # option that takes a value

    # Number of environments (the outer loop)
    parser.add_argument('--outer', '-outer', type=int, default=10)  # outer loop

    # Number of days for a given (slightly changing) environment (and a given agent)
    # ( or 'inner' loop.. number of inner iterations ) (see outer too..)
    parser.add_argument('--days', '-days', default=20, type=int)
    # Max number of steps in a day (if <= 0, it's None or no limit)
    parser.add_argument('--max_steps', '-ms', default=0, type=int)

    # Average over the last d days for each environment (when lastd > 0).
    parser.add_argument('--lastd', '-lastd', type=int, default=0)

    # seed for random number generator
    parser.add_argument('--seed', '-seed', type=int, default=None)
    
    # the environment

    # environment/grid size (size x size)
    parser.add_argument('--size', '-size', type=int, default=15)
    
    # proportion of barriers. (barrier rate)
    parser.add_argument('--prop', '-prop', type=float, default=0.1)
    
    # change rate of barriers (for a given environ, from one day to next).
    parser.add_argument('--chr', '-chr', default=0.1, type=float)

    # motion noise (a probability) in the environment.
    parser.add_argument('--mnoise', '-noise', default=0.0, type=float)

    # if -doh then DO human mode (display, or NOT  the fast batch mode)
    parser.add_argument('--do_human', '-doh',  # not -noh any more!
                        action='store_true', default=False)  # do human mode?
    
    # Place a food at random location?
    parser.add_argument('--rf', '-rf', action='store_true', default=False)

    # Always have a target/goal defined (instead of leaving it None?)
    parser.add_argument('--atd', '-atd', action='store_true', default=False)

    # num food locations (if <= 1, do not change food locations,
    # otherwise an integer up to 4 corners): different food locations
    # on different days, but one food location per day.
    parser.add_argument('--food_locs', '-fls', default=1, type=int )
    # Round-robon change: applicable when food_locs >= 2   ( 2 to 4 locations )
    parser.add_argument('--do_rr', '-do_rr', action='store_true', default=False )

    # place food in the grid interior (not just on corners) (false by default)
    parser.add_argument('--do_interior', '-do_interior', default=False, action='store_true' )
    
    # abort when truncated ( so the user knows )
    parser.add_argument('--awt', '-awt', action='store_false', default=True)

    # Output the grid configurations in text format, before doing
    # the experiments.
    parser.add_argument('--grids', '-grids', action='store_true', default=False)

    # the strategies, comma separated, to use (to mix in the composite
    # agent)
    parser.add_argument('--sts', '-sts', default='2,1', type=str)

    # time budgets, for each strategy (if 0 for a strategy, then use
    # default)
    parser.add_argument('--budgets', '-buds', default='', type=str) 

    # agent capability.
    
    parser.add_argument(
        '--path_intgrate', '-integ', type=int, default=0
    )  # path integrate ?

    ###################################
    #
    # For different strategies..
    
    # If set to true, smell greedy is used when mixed-greedy is used.
    parser.add_argument('--smell_greedy', '-dsg', type=int, default=1)

    # Don't go back (unless you have to). Improves random
    # strategies. (skip going back ) by default its true. If -nob,
    # it's turned off..
    parser.add_argument('--noback', '-nob', action='store_false', default=True)
    
    parser.add_argument('--tick_beta', '-tbeta', type=float, default=0.4)
    parser.add_argument('--day_beta', '-dbeta', type=float, default=0.1)
    parser.add_argument('--perception_conf', '-pconf', type=float, default=0.9)
    # memory: too_old_for_use
    parser.add_argument('--tofu', '-tofu', type=int, default=-1)

    # two level updating?
    # parser.add_argument('--bilevels', '-2levs', type=int, default=1)

    # If set to true, the prob-map (map-memory) agent will replan when
    # it hits a wall, or time expires.
    # parser.add_argument('--replan', '-replan', type=int, default=1)
    
    args = parser.parse_args()
    return args

###


########


#
# Try one agent type (strategy) from misc_agents file in an
# environment for several days (after obtaining the reward or food,
# next day begins and the environment, ie barrier locations, might
# change.. see 'fixed_food' below). Try the agent for n_outer many
# times, each time start with a new environment, and the agent type is
# initialized as well (anything learned don't carry over, so to get an
# 'outer' average over many initial environments).
#
def outer_explore_an_agent(envs_list=None, seed=None):
    args = parse_args()

    print('\n# (In outer explore) arg values:\n\n', args, '\n')
    
    start_time = time.time()
    # if args.save_display and not os.path.isdir(args.output_dir):
    #    os.makedirs(args.output_dir)
    n_outer = args.outer
    if envs_list is not None:
        n_outer = len(envs_list)

    # If something goes wrong (eg strategy returns None or too many
    # steps in a day), abort.
    abort_when_truncated = args.awt
    truncated = False
    
    means, meds, days2, days = [], [], [], []
    timeouts, maxs = [], []
    for i in range(n_outer):
        env_days = None
        if envs_list is not None:
            env_days = envs_list[i]
        print('\n# ****\n#\n# Starting outer itr:', i+1, '\n')
        res = inner_loop_explore_an_agent(outer=i+1, env_days=env_days, seed=seed)
        if res is None:
            print('# in this outer itr %d None was returned..' % i)
            if abort_when_truncated:
                truncated = True
                break
            continue
        mu = res[0]
        means.append(mu) # the mean daily steps for this environment
        meds.append(res[1]) # the median daily steps for this environment
        days.append(res[2])
        days2.append(res[3])
        timeouts.append(res[4])
        maxs.append(res[5])
        print('\n\n# ****\n#  Done with Outer itr:', i+1, ' num:',
              len(means), 'last-mean:%.1f ' % mu,
              ', mean-avg-daily:%.1f (std:%.1f)' % (np.mean(means), np.std(means)),
              ' median:%.1f' % np.median(means),
              ' maxAvg:%.1f' % np.max(means),
              ' maxMax:%.0f' % np.max(maxs) )
        sys.stdout.flush()

    if abort_when_truncated and truncated:
        print(
            ('\n\n# NOTE: Aborted!! ( all strategies failed,  or some days took too long..  ) \n'))
        return

    if len(means) > 0:
        print(
            (
                '\n\n# outer itrs: %d, num days (or episodes) per environ: %.1f '
                +
                ' avg_num_timeouts: %.1f\n'
                + '\n# avg_mean_cost: %.2f (std:%.2f, num:%d)\n# median_mean_cost: ' +
                '%.2f, max:%.2f, maxMax:%.0f, medMed:%.1f'
            )
            % (
                n_outer,
                # args.T, np.mean(steps), np.min(steps),
                np.mean(days),
                np.mean(timeouts),
                np.mean(means),
                np.std(means),
                len(means),
                np.median(means),
                np.max(means),
                np.max(maxs),
                np.median(meds)
            )
        )
    print('\n')
    if np.mean(timeouts) > 0:
        print('\n# NOTE: ***** there were %d timeout days.. (out of %.1f days) ***\n  ' %
              (np.sum(timeouts), n_outer * np.mean(days) ))

    print('# Time taken: %.2f mins\n' % get_mins_from(start_time))

######


# return a list of integers (corresponding to strategies).
def extract_strategies(strategies_str):
    return [ int(x) for x in strategies_str.split(',') ]

def extract_budgets(budgets_str, sts):
    if budgets_str == '':
        return []
    buds = [ int(x) for x in budgets_str.split(',') ]
    # no longer than length of strategies.
    buds = buds[:len(sts)]
    return buds
    
def set_strategy_params(args):
    StrategyParams.do_smell_greedy = args.smell_greedy
    StrategyParams.always_remove_backward = args.noback # do biased random?
    # for the agent, generic (not strategy specific)
    StrategyParams.use_path_integration = args.path_intgrate

    # for prob-map (for experimenting)
    StrategyParams.day_beta = args.day_beta
    StrategyParams.tick_beta = args.tick_beta
    StrategyParams.perception_conf = args.perception_conf
    StrategyParams.too_old_for_use = args.tofu
    # StrategyParams.two_level_probs = args.bilevels

#
#
# Start with an initial environment, and repeat (trying to get to
# food) for several days (the list env_days contains the environment,
# the grid configuration, for each day). The environment may change
# slightly from one day to next (location of barriers, etc). Report
# and return the average number of steps to food, etc.
def inner_loop_explore_an_agent(fixed_food=True, outer=None,
                           env_days=None, seed=None):
    # too_many_steps = 100000  # some high number (set below too).
    human_mode = 'None'  # render mode
    if args.do_human:
        human_mode = 'human' # enable human mode

    #T = args.T  # Repeat up to this many time steps.
    #if T == 0:
    #    T = None  # total steps **across** days

    # Either put a high cost when truncation occurs or abort immediately.
    abort_when_truncated = args.awt
    max_cost = 100000
    
    max_steps = None  # Within a day
    if args.max_steps > 0:
        max_steps = args.max_steps

    #if env_days is None:
    if env_days is not None:
        env = env_days[0]
        num_days = len(env_days)
    else:
        # max number of days to gather stats
        num_days = int( args.days )
        size = int(args.size)
        if fixed_food:  # One food?
            env = BarriersEnv(
                render_mode=human_mode,
                size=size,
                prop=float(args.prop),
                change_rate=float(args.chr),
                place_at_random=args.rf,
                multiple_foods=False,
                motion_noise=args.mnoise,
            )
            env = EnvUtils.get_env_with_a_path(env)
            if env is None:  # no path was found? exit.
                # exit(0)
                # return 0, 0, 0, 0, 0
                return None

        else:  # choose multiple foods at random locations?
            env = BarriersEnv(
                render_mode='human',
                size=size,
                prop=float(args.prop),
                place_at_random=args.rf,
                multiple_foods=1,
                motion_noise=args.mnoise,
            )

    #if not args.old: # Allocate the agent, use the new composite
    sts = extract_strategies(args.sts)
    agent = CompositeAgent(env, sts, seed=seed)
    agent.set_budgets(extract_budgets(args.budgets, sts))
    set_strategy_params(args)
        

    observation, info = env.reset(0)  # seed = 42)
    tot, reward = 0, 0  # num foods attained so far, etc..

    timeouts, truncated, cost, terminated = 0, False, 0, 0
    # costs is number of steps in each day.
    costs, total_steps = [], 0
    days = 1  # number of days
    while days <= num_days:
        if env_days is not None:
            env = env_days[days-1]
            agent.set_env(env)
            
        cost += 1  # cost could be reset (begining of a day)
        total_steps += 1  # this is not reset.
        truncated = False
        # Get the next action to execute.
        #if not args.old:
        # User-defined agent function.
        action = agent.get_action(  ) # observation, reward )
        #else:
        #    action = agent.execute(
        #        observation, reward, day=days+1, outer=outer)  # User-defined agent function
        
        if action is None or (max_steps is not None and cost >= max_steps):
            reward = 0 # set 0, so if action is None, reward from prev itr
            # is not used..
            if max_steps is not None:
                costs.append(max_steps)
            elif max_cost is not None:
                costs.append(max_cost)
            print(
                (
                    '\n# no more action possible or agent has given up or'
                    + ' times up.. exiting/resetting at time: (cost=%d, day=%d)'
                )
                % (cost, days+1)
            )
            if fixed_food:
                truncated = True
                cost = 0
            else:
                break
        else: # execute the action..
            observation, reward, terminated, truncated, info = env.step(action)

        #if args.save_display:
        #    fn = os.path.join(args.output_dir, f'image{i:04d}.png')
        #    Image.fromarray(env.rgb_array).save(fn, 'PNG')

        if truncated and abort_when_truncated:
            break
        
        # Inform the agent! (that a reward was found, eg to remember
        # the location of food for next day).
        if reward > 0:
            print('# OBTAINED food! cost: %d steps' % cost)
            agent.upon_obtained_reward(reward)
            # if reward > 0:
            # print(f'reward found = {reward}')
            tot += reward
            #agent.reset_for_new_day(truncated) # this is done below
            costs.append(cost)
            print('\n\n# ====>  at days=%d.. total reward: %d cost:%d' % (days, tot, cost))
            print(('\n\n# ==> end of day=%d env=%s.. mean daily ' +
                  'steps: %.1f, median: %.1f, max: %d') % (
                      days, str(outer), np.mean(costs), np.median(costs),
                      np.max(costs) ))
            cost = 0
            print(
                '# ==> reward per step: %.3f (or num steps to reward %.1f)\n\n'
                % (tot / (days+1.0), (days+1.0) / tot)
            )
            # double checking my assumption, one of these should be
            # true!
            assert terminated or truncated
            if args.do_human:  # if human display mode, sleep (slow down!).
                time.sleep(2)
                
        # A reward was found, or timed out.
        if terminated or truncated:
            if env_days is  None:
                env.increment_day()
                _, _ = env.reset()

            days += 1
            # print('# day is now:', env.get_day())
            # print('# agents day: ', agent.env.get_day())

            if truncated:
                print('# TRUNCATED!')
                timeouts += 1

            if fixed_food and days <= num_days and env_days is  None:
                env.change_env_map(env=env)  # change map here
                # Is there a path? try creating a new one till you get a path.
                env = EnvUtils.get_env_with_a_path(env)
                if env is None:  # give up?
                    # exit(0)
                    print('\n# stopping/breaking the loop (giving up)..\n')
                    break

            if days <= num_days:
                # reset some aspects of the agent too..
                agent.prepare_for_new_day(truncated)

                print('\n\n# ====> Resetting/preparing..  for next day=%d..\n\n' % days)
                if args.do_human:  # if  display mode, sleep (slow down!).
                    time.sleep(2)
            
            if days > num_days:
                days -= 1
                break

    if truncated and abort_when_truncated:
        return None
    
    # Final print out of number times food was found within alloted
    # time T, etc.
    if days > 0 and tot > 0:
        print('\n# Done at day=%d.. total reward: %d' % (
            days, tot))
        #print(
        #    '# reward per step: %.3f (or avg num steps to reward %.1f)\n'
        #    % (tot / total_steps, total_steps / tot)
        #)
    else:
        print('\n# Done at day=%d.. total reward: %d\n' % (days, tot))

    print('# 1st few costs (steps):', costs[:20])
    if len(costs) > 0:
        print(
            (
                '\n# days:%d  num (successes plus timeouts):%d '
                + '(timeouts:%d), mean cost per days:%.2f, median %.2f, max:%.2f'
            )
            % (
                # T,
                days,
                len(costs),
                timeouts,
                np.mean(costs),
                np.median(costs),
                np.max(costs),
            )
        )

    if len(costs) > 0:
        if args.lastd <= 0:
            mu = np.mean(costs)
        else:  # average over the last few days only (assuming good
            # convergence)
            mu = np.mean(costs[-args.lastd :])
            # For all stats computed, do it on this subset (eg last 10 days).
            costs = costs[-args.lastd : ]
        # return mean, median (over the day) and map, etc
        return mu, np.median(costs), days, len(costs), timeouts, np.max(costs)
    return None


# >end

###

if __name__ == '__main__':
    # To test 'misc agents' on misc. environments!
    args = parse_args()

    if 1:
        print('\n\n# the args are:', args)
    envs_list = None
    if 1:
        envs_list = make_envs.make_environments(args)
        if args.grids: # print out the environments/days in text format
            print('\n# num environments:', len(envs_list))
            i = 0
            for envs_days in envs_list:
                i += 1
                print('\n-----\n')
                for env in envs_days:
                    #else:
                    print('\nday:%d dim:%d b:%d f:%d (env:%d)' % (
                        env.get_day(), env.get_size(), env.get_barrier_count(),
                        env.get_food_count(), i ))
                    if 1:
                        #print('->day:%d\n%s' % (env.get_day(), env.barriers_to_text()))
                        print('\n--> day %d grid:\n%s' % (env.get_day(), env.barriers_to_text()))

    
    outer_explore_an_agent(envs_list, seed=args.seed)
    exit(0)
    # env.close()
