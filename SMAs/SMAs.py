#
#
# This file, following the paper 'Tracking Changing Probabilities via
# Dynamic Learners' (https://arxiv.org/abs/2402.10142v3), includes a number 
# of sparse moving average techniques ('SMAs'), such as the Sparse EMA, a rate-based
# technique, and count-based techniques (or queue
# based, eg Qs), and a combination called DYAL. 

#
# SMAs are designed for (detecting and) tracking changing probabilities, i.e., tracking
# proportions of (discrete) items over a (possibly unbounded) non-stationary 
# stream of items. For the current techniques below, 'detection' of change, is implicit. 

# DistroQs was written for Flatland barriers experiments.

# The SMAs in this file as of this writing: sparse EMA ('Sima'), DYAL, Qs,
# DistroQs, Box, and TimeStampQs.

# common abbreviations used below:
#
# distro: distribution or semi-distribution
# SD or sd: semi-distribution
# lr: learning rate  (and related:  minlr, maxlr, etc)
# obs : an observation (an item)
# prob or pr: probability
# Qs: The 'queues' technique
#
#


from collections import Counter, defaultdict
import random, numpy as np, sys, math

from math import log

from flatland.SMAs.sma_utils import kl_bounded

###

# Decays (lowers) the given rate and returns the result (the lowered
# value).  Does the lowering via the 'harmonic way' (a double
# reciprocal). So an lr of 1/2 becomes 1/3, lr of 1/3 becomes 1/4, and
# so on.
#
# Assumes the given lr is positive in (0, 1].
def harmonic_decay(lr, min_lr=0):
    lr = 1.0 / ( (1.0/lr) + 1.0 )
    lr = max(lr, min_lr)
    return lr

##

# Update an entry (obs) given an SD (distro).
def plain_ema_update(obs, distro, rate):
    if obs not in distro:
        distro[obs] = 0
    for c, w in distro.items():
        w *= (1.0 - rate)
        if c == obs:
            w += rate
        distro[c] = w

###

# Common constants/options for SMA classes.
#
#
# >class >smac
class SMA_Constants:
    # If an item has prob below this thresh, it can be
    # ignored/dropped.  This is the minimum probability supported (eg
    # it could be 0.01, 0.001, or 0.0001)
    min_prob = 0.01

    # Instead of 1.0, always leave some slack for EMA weights.
    # NOTE: not all EMA variants may abide by this.
    max_ema_sum = 0.999  # dyal is.
    
    # starting point of lr (for EMA harmonic decay)
    max_harmonic_lr = 0.1 # could be 1, 0.1, etc
    
    # Drop low PRs from EMA probs map within dyal (but not
    # from queues) (if other conditions met...)
    drop_below_min_prob =  True # 1

    min_lr = 0.0005 #  min learning rate
    
    # For users of queues.
    q_capacity = 3
    # These are used in the binomial test of figuring out whether
    # to switch/listen to the queues.
    # thrsh of 3 corresponds to 95% confidence, 5 to >= 99% confidence,
    # 7 is >= 99.9% (and 2 is 87% or almost 90%)...
    binomial_thrsh = 3

    @classmethod
    def print_settings(cls ):
        print('\n # .. SMA_constants, min_lr: %.4f' % cls.min_lr)
        print('   # .....  q capacity: %d, binom thrsh:%.1f' % (
            (cls.q_capacity, cls.binomial_thrsh )))
        print('   # .... drop below min prob:%d' % (
            (           cls.drop_below_min_prob)))
        print()


###########

# A single queue of counts for a single observation (obs), with a
# constant maximum integer capacity, ie number of cells in the queue
# (imagined to be small, eg 3 or 5, or 10 etc.). Used in Qs and DYAL.
# Each cell in the queue keeps a count. See also TimeStampQ (similar
# in several ways).  >class
class ObsQ: # Queue of observation counts.
    def __init__(self, capacity=None):
        # NOTE: queue[0] is the most recent cell, and it's
        # under-construction (or partial, incomplete), or its estimate
        # is not trust-worthy (a poor upperbound) (however, if it
        # turns out to be lower than the others, it's a valuable
        # signal..).. the estimate of each of the other (completed)
        # cells are proper upperbound (their estimate, 1/count,
        # remains biased and an upper bound)..
        self.queue = []
        # Note with current set up, capacity should be 2 or higher to
        # obtain positive probabilities.
        self.k = capacity
        if self.k is None:
            self.k = SMA_Constants.q_capacity
            
        # We derive an equivalent of learning rate from the
        # queue counts
        self.rate_smoother = 1

    ####
    
    # In obsQ .. NOTE (aug 2023: goes over *all* cells): returns 3
    # values, "upper", "lower", "unbiased", but all could be upper
    # bounds in expectation, since it uses the newest/back cell of the
    # queue too, which is a partial or incomplete cell.  Still, if we
    # ignore this cell, we get a higher variance estimate based on
    # experiments, testing in probabilities of range 0.01 to 1.0.. so
    # I rather take the 'lower' or 'unbiased' estimate from this
    # method (which still could be an uppder bound, vs completely
    # ignoring the partial back cell.. )
    #
    # NOTES: the lower prob is technically not always lower estimate
    # (and other ones are also not sound.. or a bit higher
    # estimates!), since here we are also using the partial
    # (most-recent) queue cell (back of the queue).  If we only look
    # at the other (complete) cells, then I conjecture that
    # subtracting one from that count (in the numerator), would yield
    # a proper lower bound.  HOWEVER: for non-stationarity, specially
    # if the back partial cell of the queue has a lower prob (than
    # other complete queue cells/bins), it should be used..  (imagine
    # if the item suddenly disappears) the partial bin should not be
    # ignored ...
    def get_upper_lower_probs_all(self):
        count = self.get_count()
        if count <= 0:
            return 0, 0, 0
        numerator = 1.0 * len(self.queue)
        if count <= 1.0:
            return numerator / count, 0.0, 0.0
        return numerator / count, (numerator-1) / count, \
             (numerator-1) / (count-1.0)

    # In obsQ .. Goes over the 'completed' queue cells only (all cells
    # except the back or the newest cell of the queue), to get proper
    # lower and upper bounds. However, in experiments (eg in Aug
    # 2023), this method yields a higher variance estimate compared to
    # the method that goes over all the cells (the one that includes
    # the back cell too). Furthermore, does not handle
    # non-stationarity (eg sudden changes).
    #
    # Use "get_upper_lower_probs_all()".
    def get_upper_lower_probs_completed(self):
        nc = len(self.queue) - 1.0 # All except the back cell.
        if nc <= 0: # numerator is 0.
            return 0.0, 0.0, 0.0
        count = self.get_completed_count()
        if count <= 0:
            return 0.0, 0.0, 0.0
        if count <= 1:
            return nc / count, 0.0, 0.0
        return nc / count, (nc - 1.0) / count, \
            (nc - 1.0) / (count - 1.0) # unbiased, MVUE

    ####
    
    # In ObsQ (a single queue)
    #
    # NOTE: Note with current set up, queue capacity should be 2 or
    # higher to obtain positive probabilities.
    def get_prob(self):
        # upper, lower, unbiased = self.get_upper_lower_probs_completed()
        #
        # See comments for get_upper_lower_probs_all.
        upper, lower, inbetween = self.get_upper_lower_probs_all( )
        return inbetween

    # Get the count in the back cell (most recent cell) of the queue.
    def get_back_count(self):
        return self.queue[0]
    
    # Get estimate (reciprocal) from back of the queue
    def get_back_reciprocal(self):
        return 1.0 / self.queue[0]

    # Get the equivalence of a learning rate (lr) from the queue.
    def get_qrate(self):
        count = self.get_count()
        return 1.0 / (count  + self.rate_smoother)

    # in ObsQ (skips the back cell in computing total count).  Compute
    # the total count over completed cells only.
    def get_completed_count(self, i=None):
        return sum(self.queue) - self.queue[0]

    # in ObsQ (total count over all cells)
    def get_count(self, i=None):
        # NOTE: this works on an empty queue as well. Returns 0 in
        # that case.
        return sum(self.queue)
    
    # in ObsQ..
    # For now, returns the probability, the count, and num queue
    # cells (used in simplified update in ema+dyal).
    def get_qinfo(self):
        prob = self.get_prob()
        return prob, self.get_count(), len(self.queue)
    
    # just the num queue cells
    def get_pos_count(self):
        return len(self.queue)

    # return total count over all cell, and num cells
    def get_counts(self, i=None):
        if i is None:
            return sum(self.queue), len(self.queue)
        return self.queue[i], 1

    def get_contents_str(self):
        cstr = ''
        for c in self.queue:
            cstr += ' %d' % c
        return '[' + cstr + ' ]'
    
    def is_full(self):
        return len(self.queue) == self.k

    # Increment the back of the queue, ie a 'negative' update.
    def increment_count(self):
        self.negative_update()
        
    def negative_update(self):
        # The first is the most recent, and its estimate
        # is under-construction (and not trustworthy)..
        if self.queue == []:
            # If empty.. (shouldn't happen in current clients of
            # ObsQ..).. then no operation (no change).
            return
        self.queue[0] += 1

    def insert_obs(self): # a positive update.
        self.positive_update()

    # Handle a positive observation (which adds an entry in back
    # removes one in the front of queue, if need be..), a 'positive'
    # update.
    def positive_update(self): # a positive update.
        self.queue = [1] + self.queue
        self.queue = self.queue[:self.k]

    # (in ObsQ  )
    # 
    # Allocate a new queue and copy contents from queue q.
    @classmethod
    def copy_q(cls, q):
        #assert len(q.queue) > 2
        q2 = ObsQ(q.k)
        for c in q.queue:
            q2.queue.append(c)
        return q2

#####

# >Qs or the 'queues' technique.
#
# The idea of using (count) queues.
# (Instead of using a combo of EMA and queues, etc)
# >class pureQ    >queues ,  class queues
class Qs:
    def __init__(self, q_capacity=None, min_prob=None):

        self.q_capacity = q_capacity
        if self.q_capacity is None:
            self.q_capacity = SMA_Constants.q_capacity

        self.min_prob = min_prob
        if self.min_prob is None:
            self.min_prob = SMA_Constants.min_prob

        # SMAs handle probabilities down to some minimum positive
        # probability.
        assert self.min_prob > 0

        # If a map size is below this, don't drop/prune (for EMA, DYAL and
        # Qs).  (some multiple of minimum probability supported, eg 1.5x)
        self.max_map_entries = int(1.5 / self.min_prob)
        # Check the map size (for pruning) every so often.
        self.prune_schedule = 1.5 * self.max_map_entries
        self.updates_since_last_check = 0
        
        # Informational only.
        self.update_count = 0
        
        self.reset() # allocate and reset all maps

    def reset(self):
        # map of concept/item to queue
        self.c_to_q = {}

    def get_sizes(self):
        return [ len(self.c_to_q) ]

    def get_items(self):
        return list(self.c_to_q.keys())

    def get_qstats(self, obs, contents=0):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return str(0)
        n, pos = q.get_counts()
        cstr = ''
        if contents:
            cstr = q.get_contents_str()
        return 'ncells:%d tot:%d pos:%d %s' % (
            q.get_pos_count(), n, pos, cstr)

    def get_name(self):
        return "Qs each-q capacity=%d, prune schedule:%d" % (
            self.q_capacity,  self.prune_schedule)

    # in Qs
    def predict_and_update(self, obs, t=None ):
        prob = self.get_prob(obs)
        self.update(obs)
        return prob

    def get_counts(self, obs):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return 0, 0
        count = q.get_count()
        pos_count = q.get_pos_count()
        return count, pos_count

    # in Qs SMA
    def get_upper_lower_probs(self, obs):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return None, None, None
        return q.get_upper_lower_probs()
    
    # Get (possibly normed) prediction probability in Qs.
    def get_prob(self, obs):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return 0 # None
        return q.get_prob()

    # in Qs
    def update(self, obs):
        self.update_count += 1
        self.updates_since_last_check += 1 
        if self.updates_since_last_check > self.prune_schedule:
            self.prune_map()
            
        q = self.c_to_q.get(obs, None)
        if q is None:
            q = ObsQ(self.q_capacity)
            self.c_to_q[obs] = q
        q.insert_obs()
        # 'weaken' the rest (increment their q count)..
        for c, q in self.c_to_q.items():
            if c == obs:
                continue
            q.increment_count()

    # in Qs
    def get_sump(self):
        sump = 0
        for c, q2 in self.c_to_q.items():
            prob2 = q2.get_prob()
            if prob2 is None:
                continue
            sump += prob2
        return sump
    ##

    # (used for prunning, in Qs) Returns the set of items after kth
    # lowest item and/or those with probability below tiny pr.  NOTE:
    # dropping below a tiny pr is supported (even if the map may
    # otherwise have a relatively small number of entries, ie not too
    # big to prune), so to keep the space consumption by individual
    # PRs and counts under control (may be relevant to processing
    # long/infinite streams).  Set it to 0 if you don't want dropping
    # by such a criterion.
    @classmethod
    def compute_removables(cls, c_to_q, k, tiny_prob=0):
        c_to_pr = {}
        for c, q in c_to_q.items():
            # use all counts
            n, pos = q.get_counts()
            c_to_pr[c] = 1.0 * pos / n
        return cls.lowest_prs(c_to_pr, k, tiny_prob=tiny_prob)

    # in Qs: return the set of low(est) PR items union items with PR
    # below tiny_prob threshold (lowe PR items).
    @classmethod
    def lowest_prs(cls, c_to_pr, k, tiny_prob=0):
        pairs = list(c_to_pr.items())
        # make the set of those below with PR below tiny_prob
        tiny_remove = set([
            x[0] for x in filter(lambda x: x[1] < tiny_prob, pairs)])
        pairs.sort(key=lambda x: -x[1]) # decreasing order
        #
        # Items after kth onwards are returned (union it with
        # tiny_remove)
        return tiny_remove.union(set([x[0] for x in pairs[k:]]))

    # Prune map if it is too big, or those items with very low
    # PR (square of min_prob here).
    def prune_map(self):
        self.updates_since_last_check = 0 # reset
        # tiny prob is defined as square of min_prob.
        tiny = self.min_prob * self.min_prob
        to_remove = Qs.compute_removables(
            self.c_to_q, self.max_map_entries, tiny)
        # print('# tiny and to_remove:', tiny, to_remove)
        for c in to_remove:
            self.remove_item(c)    
        
    def set_prune_schedule(self, thrsh):
        self.prune_schedule = thrsh

    def remove_item(self, c):
        self.c_to_q.pop(c)

    ##

    # NOTE1: each value in the map returned will be a probability, but
    # may violate the SD property, ie the sum of the values can exceed
    # 1.0, unless normalize is set to true (in that case, will add to
    # 1). NOTE2: can return an empty SD as well (when no item, or all
    # PRs are 0).
    def get_distro(self, normalize=0):
        sump = 0.0
        if normalize:
            sump = self.get_sump()
            if sump <= 0.0:
                return {}
        distro = {}
        for c, q in self.c_to_q.items():
            prob = q.get_prob()
            if prob is None:
                continue
            if normalize:
                distro[c] = prob / sump
            elif prob > 0:
                distro[c] = prob
        return distro
    
    def get_raw_prob_map(self):
        return self.get_distro(normalize=0)
    
    # in Qs
    def get_weight_map(self):
        # we  need to construct the map..
        c_to_p = {}
        for c, q in self.c_to_q.items():
             c_to_p[c] = q.get_prob()
        return c_to_p

    # get the effective learning rate for the edge to c.
    def get_lr(self, c):
        # assert False
        q = self.c_to_q.get(c, None)
        if q is None:
            return 0.0
        return 1.0 / q.get_count()

    def get_edge_pos_update_count(self, item, use_queue=1):
        q = self.c_to_q.get(item, None)
        if q is None:
            return 0
        return q.get_pos_count() # just the num queue cells

####

# the 'box' predictor or fixed-window estimator: has O(1) time for
# update (if implemented efficiently)..  Keeps a single queue.. of 100
# or 1000 (or 100s to 1000s) of item..  So its space consumption is
# fixed and rigid, and not data driven (a potentially huge
# disadvantage...) its PRs form a proper DI (distribution), so that's
# a pro.. but if we want to be reactive to changes, we want a small
# history, while if we want to support small probs, eg 0.01 or 0.001,
# we want a long history..  So there is tradeoff here...  >class
class BoxCell: # a queue cell in the box
    def __init__(self):
        self.next = None
        self.prev = None
        self.item = None

# >class >box 
class Box:
    # Keeps a single queue.
    def __init__(self, capacity=100):
        self.capacity = capacity # capacity or (the fixed) window size.
        self.reset()
        
    def reset(self):
        self.cmap = Counter() # The count map
        self.latest = None # The latest cell
        self.last = None # The last (queue) cell
        self.size = 0

    def update(self, obs):
        cell = BoxCell()
        cell.next = self.latest
        cell.item = obs
        self.cmap[obs] += 1
        if cell.next is None:
            assert(self.size == 0)
            self.last = cell
            self.size = 1
            self.latest = cell
            return
        cell.next.prev = cell
        self.latest = cell
        if self.size >= self.capacity: # drop the last queue cell
            item = self.last.item
            self.cmap[item] -= 1 # and drop item too if 0.
            if self.cmap[item] == 0: # and drop item too if 0.
                self.cmap.pop(item)
            # Now set the new (or queue front)
            self.last = self.last.prev
            self.last.next = None
        else:
            self.size += 1

    def get_prob(self, obs):
        if self.size == 0:
            return 0
        c = self.cmap.get(obs, 0)
        if c == 0:
            return 0
        return 1.0 * c / self.size

    def get_distro(self):
        sd = {}
        for item, c in self.cmap.items():
            sd[item] = 1.0 * c / self.size
        return sd

    # In the Box method
    def predict_and_update(self, obs ):
        prob = self.get_prob(obs)
        self.update(obs)
        return prob

    # provide
    def get_count(self, obs):
        return self.cmap.get(obs, 0)

    def get_name(self):
        return ("Box predictor with window size:%d") % (
            self.capacity)
   
#####

# the time-stamp idea
#
# NOTE: this is only a *single* queue for the time stamps method
# ... see TimeStampQs (time-stamp queues) further below for the
# 'whole' TimeStampQs method ...
#
# (counts/gaps can be derived from an internal clock).. Leads to more
# efficient updates. Useful for computing (moving) priors, etc..
# >class TimeStampQ
class TimeStampQ:  # single queue
    def __init__(self, capacity=5, use_plain_biased=False):
        # NOTE: queue[0] is the most recent cell, and it's
        # under-construction (partial), or its estimate is not
        # trust-worthy (the least trust worthy cell!)..
        self.queue = []
        self.k = capacity
        # Use the first, incomplete cell too? (for fast turn around?)
        self.use_plain_biased = use_plain_biased
        # to support fractional values too
        self.prop_q = [] # proportion queue (numerators, if not 1)

    # NOTE: here, by default, we will ignore the most recent cell.  We
    # will only consider the older completed cells. However, if the
    # most recent cell has an estimate that's actually lower than
    # average (of the completed cells), then use it too!
    def get_prob_proportions(self, tnow, same_earliest_time=None):
        if self.use_plain_biased:
            return self.get_prob_props_plain(tnow, same_earliest_time)
        qlen = len(self.queue)
        if qlen <= 1:
            return 0.0
        assert len(self.prop_q) == qlen
        # count or time stamp of rest (ie completed cells)
        c_rest = self.queue[0] - self.queue[-1]
        assert c_rest > 0, 'c_rest:%s qlen:%d tnow:%d q(0):%d q(1):%d' % (
            str(c_rest), qlen, tnow, self.queue[0], self.queue[1])
        
        p1 = sum(self.prop_q[1:]) / c_rest
        c0 = tnow - self.queue[0] + 1.0
        assert c0 > 0, 'c0:%d qlen:%d tnow:%d q(0):%d q(1):%d' % (
            c0, qlen, tnow, self.queue[0], self.queue[1])
        p2 = self.prop_q[0] / c0
        if p2 < p1: # possibly return p2 (non-stationarity)
            # but only when the count is large enough..
            if c0 > 2.0 * c_rest / (qlen-1):
                return p2
        return p1

    # We won't ignore first cell: for fast returns.
    def get_prob_props_plain(self, tnow, same_earliest_time=None):
        qlen = len(self.queue)
        if qlen <= 0:
            return 0.0
        assert len(self.prop_q) == qlen
        # count or time stamp of rest (ie completed cells)

        if same_earliest_time is not None:
            c_all = tnow - same_earliest_time + 1.0
        else:
            c_all = tnow - self.queue[-1] + 1.0
        return sum(self.prop_q) / c_all
        #return p1

    def get_earliest_time(self):
        return self.queue[-1]

    #######

    # Only uses completedd cells.
    # In TimeStampQ (single queue). tnow is time now.
    def get_completed_proportions(self, tnow):
        qlen = len(self.queue)
        if qlen <= 1:
            return 0.0
        assert len(self.prop_q) == qlen
        # count or time stamp of rest (completed cells)
        c_rest = self.queue[0] - self.queue[-1]
        return sum(self.prop_q[1:]) / c_rest

    # In TimeStampQ (single queue). tnow is time now.
    def get_completed_prob(self, tnow):
        if self.prop_q != []:
            return self.get_completed_proportions(tnow)
        qlen = len(self.queue)
        if qlen <= 2: # too few.. (we need at least 2 completed cells
                      # and one partial, or 3 cells in the queue.)
            return 0.0
        # completed cells only.
        p1 = (qlen - 2.0) / (self.queue[0] - self.queue[-1] + 1 - 2 )
        return p1

    ########
    
    # In TimeStampQ (single queue). tnow is time now.
    def get_prob(self, tnow, same_earliest_time=None):
        if self.prop_q != []:
            return self.get_prob_proportions(tnow, same_earliest_time)

        qlen = len(self.queue)
        if qlen <= 2: # too few.. (we need at least 2 completed cells
                      # and one partial, or 3 cells in the queue.)
            return 0.0
        # This is the unbiased estimation ( except the latest or 
        # cell0 is partial.. ) .. (the count in the denominator is
        # number of updates including latest time tnow.. )
        #
        # An unbiased estimate is: (count positives - 1) / (count
        # trials - 2) (where trials is up to and including the
        # observed positive) (and we do this for completed cells only,
        # or count positives is qlen-1, and note that: count trials,
        # for the completed cells, is self.queue[0] - self.queue[-1] +
        # 1)
        p1 = (qlen - 2.0) / (self.queue[0] - self.queue[-1] + 1 - 2 )
        # The latest partial cell: if p2 is less than p1, then use p2
        # (but in general, during stability or stationarity, the
        # estimate p2 is higher than the unbiased p1)
        p2 = 1.0 / (tnow - self.queue[0] + 1.0)
        return min(p1, p2)
    
    def get_info(self):
        last_t = None
        if self.queue != []:
            last_t = self.queue[0]
        return 'num queue bins: %d, last_time_pos_update:%s' % (
            len(self.queue),  str(last_t) )
    
    # tnow is time now (or latest update count).
    def get_counts(self, tnow, qlen):
        sumc = 0 # total count
        if qlen < 2:
            return 0, 0
        # tnow > queue[0] > .. > queue[i-1] > queue[i] > ..
        for i in range(1, qlen):
            #sumc += tnow - queue[i]
            sumc += self.queue[i-1] - self.queue[i]
        avgc = 1.0 * sumc / (qlen-1)
        # use the most recent one too, when it
        # yields a smaller prob..
        new_count = tnow - self.queue[0]
        # this means the estimate from newest will be less than
        # older completed cell. If so, use it too!!
        if new_count > avgc:
            # Include the latest partial cell too.
            sumc += new_count
            # The numerator for computing probabilities.
            included = qlen 
        else:
            # DON'T include the latest partial cell.
            included = qlen - 1
        return sumc, included

    # Just the num queue cells, but, by default, ignoring the partial
    # (most recent) queue cell.
    def get_pos_count(self):
        return len(self.queue) - 1

    # in TimeStampQ
    # get_size()
    def size(self):
        return len(self.queue)

    # update, using tnow (in TimeStampQ)
    def update(self, tnow, prop=None):
        self.insert_obs(tnow, prop)

    # (in time-stamp single queue) Inserts a positive observation,
    # here meaning the latest time-stamp (which adds an entry in
    # front, removes one in the back if at capacity). If tnow is
    # already there, won't add it...
    def insert_obs(self, tnow, prop=None):
        if self.queue == []:
            self.queue = [tnow]
        elif self.queue[0] != [tnow]:
            # we don't want repeat times (tnow == self.queue[0]), nor
            # new time being before pass times (tnow <
            # self.queue[0])!!
            if tnow <= self.queue[0]:
                return
            self.queue = [tnow] + self.queue
        # Keep q capacity within k
        self.queue = self.queue[:self.k]
        if prop is not None:
            # NOTE: we assume the proportion is positive.
            self.prop_q  = [prop] + self.prop_q
            # Keep q capacity (of proportions) within k
            self.prop_q = self.prop_q[:self.k]
            
######

# This one uses object of type time-stamp-queue.

# >class (the time-stamp idea for queues)
#
class TimeStampQs: 
    def __init__(self, q_capacity=5,
                 with_proportions=False,
                 use_plain_biased=False,
                 do_same_start_time=False):
        self.q_capacity = q_capacity
        self.tnow = 0 # current time or clock.
        self.reset()
        # Support for updates with fractional values (proportions,
        # instead of just 0 and 1).
        self.with_props = with_proportions
        self.use_plain_biased = use_plain_biased
        # Use same start-time (earliest) for uniform averaging
        # of the queues (plain distribution averaging).
        self.do_same_start_time = do_same_start_time
        # print("# in TimeStampQs",  self.do_same_start_time)
        
    def reset(self):
        self.c_to_q = {} # map of concept/item to queue

    def get_sizes(self):
        return [ len(self.c_to_q) ]

    def get_qstats(self, obs, contents=0):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return str(0)
        n, pos = q.get_counts()
        cstr = ''
        if contents:
            cstr = q.get_contents_str()
        return 'ncells:%d tot:%d pos:%d %s' % (
            q.get_pos_count(), n, pos, cstr)

    def get_name(self):
        return "TimeStampQs  each-q capacity=%d " % (
            self.q_capacity )
                
    # In TimeStampQs
    def predict_and_update(self, obs, prop=None, t=None ):
        prob = self.get_prob(obs)
        self.update(obs, prop)
        return prob
    
    # in TimeStampQs
    def get_upper_lower_probs(self, obs):
        q = self.c_to_q.get(obs, None)
        if q is None:
            return None, None
        return q.get_upper_lower_probs()

    def get_prob(self, obs):
        # Use same start-time (earliest) for uniform averaging
        # of queues.
        start_time = None
        if self.do_same_start_time:
            for q in self.c_to_q.values():
                est = q.get_earliest_time()
                if start_time is None or est < start_time:
                    start_time = est
        #print('# HERE88, start_time is:', start_time)
        q = self.c_to_q.get(obs, None)
        if q is None:
            # return None
            return 0 # 
        return q.get_prob(self.tnow, start_time)

    # in TimeStampQs
    def get_distro(self):
        pr_map = {}
        #for c, q in self.c_to_q.items():
        #    p = q.get_prob(self.tnow)
        for c in self.c_to_q.keys():
            p = self.get_prob(c) # So we can use flags/options.
            if p > 0:
                pr_map[c] = p
        return pr_map
    
    def get_update_count(self):
        return self.tnow
    
    # in TimeStampQs (uses several time-stamp queues)
    def update(self, obs, prop=None, update_time=1):
        if update_time:
            self.tnow += 1
        q = self.c_to_q.get(obs, None)
        if q is None:
            q = TimeStampQ(self.q_capacity,
                           use_plain_biased=self.use_plain_biased)
            self.c_to_q[obs] = q
        if self.with_props:
            # the proportion, in (0, 1], should be given.
            assert prop is not None
        else:
            prop = None # Ignore proportion
        #print(self.tnow, prop)
        q.insert_obs(self.tnow, prop)
        
    # in TimeStampQs, 'multi label' update version ( or update_multi
    # ), where the label can be proportions (values in (0,1)).  (why
    # items and props are separate, instead of list of pairs or a
    # map??)
    def distro_update(self, distro, update_time=1):
        assert self.with_props
        if update_time:
            self.tnow += 1
        for obs, prop in distro.items():
            q = self.c_to_q.get(obs, None)
            if q is None:
                q = TimeStampQ(self.q_capacity,
                               use_plain_biased=self.use_plain_biased)
                self.c_to_q[obs] = q
            q.insert_obs(self.tnow, prop)        
        # Nothing to be done for rest (no 'weakening' in the TimeStamp
        # method).

    def multi_update(self, items, props, update_time=1):
        self.distro_update(dict(zip(items, props), update_time=update_time))
        """
        assert self.with_props
        if update_time:
            self.tnow += 1
        for obs, prop in zip(items, props):
            q = self.c_to_q.get(obs, None)
            if q is None:
                q = TimeStampQ(self.q_capacity,
                               use_plain_biased=self.use_plain_biased)
                self.c_to_q[obs] = q
            q.insert_obs(self.tnow, prop)        
        # Nothing to be done for rest (no 'weakening' in the TimeStamp
        # method).
        """
                           
    def get_highest(self, k=5, only_comp=0):
        pairs = []
        for o, q in self.c_to_q.items():
            if only_comp: # only completed cells?
                p = q.get_completed_prob(self.tnow)
            else:
                p = q.get_prob(self.tnow)
            pairs.append((o,p))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

####

# >class (the distro Qs)
#
# Each update would be with a distribution, and output
# would be  a simple average of the (last k) distributions.
class DistroQs: 
    def __init__(self, q_capacity=5):
        self.queue = []
        self.k = q_capacity # num distros
        self.updates = 0 # num updates so far

    def copy(self, multiple=2): # return a copy
        c = DistroQs()
        c.queue = []
        c.k = self.k # num distros
        c.updates = 0 # num updates
        d = self.get_distro()
        # Initialize with the distro (add inertia)
        # this many times
        for _ in range(multiple):
            c.distro_update(d)
        return c

    def distro_update(self, distro):
        self.update(distro)
        
    def get_update_count(self):
        return self.updates

    def update(self, distro, make_copy=True):
        self.updates += 1
        d = distro
        if make_copy:
            d = distro.copy()
        self.queue = [ d ]  + self.queue
        self.queue = self.queue[:self.k]
        
    def get_distro(self):
        sums = Counter()
        if self.queue == []:
            return sums
        i = 0
        for d in  self.queue:
            i += 1
            for o, p in d.items():
                sums[o] += p
        # NOTE: if each entry in queue was a distribution, then
        # diving by i also creates a distribution.
        for o, p in sums.items():
            sums[o] = p / i
        return sums

    def get_prob(self, item):
        d = self.get_distro()
        return d.get( item, 0 )

####
###
#
# DYAL: supports per-edge rates, EMA together with Qs (a combination).
# 
# >class  ( >dyal >qdial )
class DYAL:

    def __init__(self, q_capacity=None, min_lr=None,
                 min_prob=None):
        self.q_capacity = q_capacity
        if self.q_capacity is None:
            self.q_capacity = SMA_Constants.q_capacity

        # This is likely deprecated! (no more need for it!)
        # self.q_smooth = q_smooth

        self.min_prob = min_prob
        if self.min_prob is None:
            self.min_prob = SMA_Constants.min_prob

        # SMAs handle probabilities down to some minimum positive
        # probability.
        assert self.min_prob > 0

        # If a map size is below this, don't drop/prune (for EMA, DYAL and
        # Qs).  (some multiple of minimum probability supported, eg 1.5x)
        self.max_map_entries = int(1.5 / self.min_prob)
        # Check the map size (for pruning) every so often.
        self.prune_schedule = 1.5 * self.max_map_entries
        self.updates_since_last_check = 0
            
        # drop small from ema probs?
        self.drop_below_min_prob = SMA_Constants.drop_below_min_prob
        
        if min_lr is None:
            self.min_lr = SMA_Constants.min_lr #
        else:
            self.min_lr = min_lr

        self.reset() # initialize all required fields/data structures

    def reset(self):
        self.c_to_prob = {} # Counter() # Map from c to the EMA probability.
        self.c_to_lr = {} # Each edge has its own EMA learning rate.
        self.c_to_q = {} # map of concept/item to queue
        # both 'negative and positive' seen counts...
        self.c_to_update = Counter()
        # this 'positive' one may be just for trouble-shooting
        self.c_to_pos_update = Counter() # update_counts or seen_times (positive updates)
        # count that at least one concept was reset to queue
        self.reset_to_q = 0
        self.update_count = 0
        # reset to bigger count (q's estimate was bigger than ema)
        self.reset_to_qbigger = 0
        # count of reset to queue, when it's consistently less.
        self.reset_to_qless = 0

    def get_name(self):
        mvc = SMA_Constants
        return ("DYAL (per-edge rates + Qs): " + 
                "qcap=%d minProb=%.3f " +
                "minLR:%.4f binoThrsh:%.0f prune_sched:%d\n") % (
                    self.q_capacity, 
                    self.min_prob, self.min_lr,
                    mvc.binomial_thrsh,
                    self.prune_schedule )

    # in dyal
    def predict_and_update(self, obs, t=None ):
        # We dont use time t, or use it for reporting only.
        prob = self.get_prob(obs)
        self.update(obs, t=t)
        return prob

    # Get prediction probability
    def get_prob(self, obs):
        prob = self.c_to_prob.get(obs, 0)
        # Nothing more to do!
        return prob

    # get both the queue prob and qrate. If queue is not allocated,
    # allocate it (and return None for qprob, etc).
    def get_qboth(self, obs, create=1):
        q = self.c_to_q.get(obs, None)
        if q is None:
            if create: # create an empty queue, if not there?
                # self.c_to_q[obs] = ObsQ(self.q_capacity, self.q_smooth)
                self.c_to_q[obs] = ObsQ(self.q_capacity)# , self.q_smooth)
            return None, None

        qprob = q.get_prob()
            
        if qprob is None or qprob == 0.0:
            return None, None
        return qprob, q.get_qrate()

    def get_sizes(self):
        return len(self.c_to_q), len(self.c_to_prob), len(self.c_to_update)
        
    # (in dyal)
    # Get the prediction probs of all (that are being predicted).
    def get_distro(self):
        return self.c_to_prob.copy()
    
    # Get the map of item to lr (the EMA lr)
    def get_lrs(self):
        return self.c_to_lr

    # Also increments all update counts entries in the map..
    def update_q_counts_etc(self, obs):
        for c, q in self.c_to_q.items():
            self.c_to_update[c] += 1
            if c == obs:
                continue
            q.increment_count()

    def harmonic_decay(self, lr):
        return harmonic_decay(lr, self.min_lr)
        
    # June 2023: should work when observation is None. (should mean
    # weaken all existing connections)
    #
    # Aug 2023. Checks the queue on every item, and reset to q_prob if
    # q_prob is lower with sufficient evidence ... if not resetting to
    # queue, then decay the rate (for all except for obs). Returns
    # available (free/unused) PR mass.
    def weaken_ema_weights_simple(self, obs):
        pairs = self.c_to_prob.items()
        used_up = 0.0 # prob mass used up
        to_drop = [] # remove these items..
        for c, ema_prob in pairs:
            if c == obs:
                used_up += ema_prob
                continue # Note lr for the target is not weakened in this function.
            assert c is not None # 'None' cannot get into this map (but obs can be None).
            q = self.get_queue(c, create=0)

            assert q is not None, 'c:%s was in ema prob map, p:%.6f, but didnt have a queue..  c_to_q:%s, in c_to_prob:%s ' % (
                c, ema_prob,  self.c_to_q.keys(), self.c_to_prob.keys() ) 
            
            qprob, qcount, _ = q.get_qinfo()
            lrc = self.c_to_lr.get(c, 0.0)
            assert lrc < 1 and lrc > 0, \
                'learning rate is: %.3f, ema_prob:%.3f, c:%s' % (
                    lrc, ema_prob, c)
            if self.drop_below_min_prob:
                if ema_prob < self.min_prob and qprob < self.min_prob:
                    to_drop.append(c)
                    continue
            if self.actual_prob_is_sufficiently_lower(ema_prob, q, qprob, qcount):
                self.reset_to_qless += 1
                self.reset_to_q += 1
                if self.drop_below_min_prob and qprob < self.min_prob:
                    to_drop.append(c)
                    continue
                self.c_to_lr[c] = 1.0 / qcount # my new learning rate.
                self.c_to_prob[c] = qprob
                used_up += qprob
            else:
                ema_prob = (1 - lrc) * ema_prob
                self.c_to_prob[c] = ema_prob
                self.c_to_lr[c] = self.harmonic_decay(lrc) # decay lrc
                #print('# In EMA weakening, plain EMA: %.3f' %  (self.c_to_lr[c]))
                used_up += ema_prob

        assert used_up < 1.0 + 0.001
        for c in to_drop:
            self.c_to_prob.pop(c)
            self.c_to_lr.pop(c)

        available = max(0, SMA_Constants.max_ema_sum - used_up)
        return available # Available in unused probability reserve.
                
    ############

    # Uses the binomial tail test.
    def whether_to_use_queue(
            self, c, ema_prob, q, qprob, qcount, check_lower=False):
        if ema_prob <= 0:
            return True
        is_higher = self.actual_prob_is_sufficiently_higher_simple(
            ema_prob, qprob, qcount)
        if is_higher:
            return True
        if not check_lower:
            return False

        lr = self.c_to_lr[c] # the current learning rate of EMA
        is_lower = self.actual_prob_is_sufficiently_lower(
            ema_prob, lr, q, qprob, qcount)
        if is_lower:
            return True
        return False

    # these can be made more elaborate, but simple comparisons worked
    # fine.
    @classmethod
    def is_qprob_sufficiently_lower(self, prob1,  qprob):
        return qprob < prob1

    @classmethod
    def is_qprob_sufficiently_higher(self, prob1, qprob):
        return prob1 < qprob

    # (actual probability as judged/estimated by the queue prob) If
    # the queue prob (proportion observed) is so large that, with very
    # high confidence, could not have been generated by the ema_prob
    # (by a coin head-prob equal to ema prob), then reset to the queue
    # prob.    
    def actual_prob_is_sufficiently_higher_simple(
            self, ema_prob, qprob, qcount):
        if not self.is_qprob_sufficiently_higher(ema_prob, qprob):
            return False
        
        # Now, check statistical significance ..
        
        # The upper binomial test: Note this is -log of upper
        # probability estimate that we see this (this many positives
        # in qcount many trials, yield the observed qprob) according
        # to the query prob (and the query prob is based on the ema
        # model).. the actual probability of this outcome happening is
        # even lower... (this is the lower end of the score, so the
        # true score is at least as high as this.. so if this score is
        # above the treshold, we should listen to the queues and
        # reject the ema prob as too unrealistic (the probability that
        # the ema_prob explains the data is too low)..
        upper = qcount * kl_bounded( qprob, ema_prob )
        return upper >= SMA_Constants.binomial_thrsh

    # Actual prob., or 'observed' prob, as judged/estimated by the
    # queue prob.  If the queue prob (proportion observed) is so small
    # that, with very high confidence, could not have been generated
    # by the ema_prob (by a coin with ema prob), then reset to the
    # queue prob.
    def actual_prob_is_sufficiently_lower(
            self, ema_prob,  q, qprob_all, qcount_all):
        # query (probability) is our model's estimate
        # query = max(ema_prob, self.min_prob)
        qcount = q.get_back_count() # back cell of the queue
        qprob = 1.0 / qcount
        if qprob_all < qprob:
            qprob = qprob_all
            qcount = qcount_all

        if not self.is_qprob_sufficiently_lower(ema_prob, qprob):
            return False
            
        # Now, check significance.
        #
        # Upper bound the chance that the query prob (our model's
        # proposed prob) yields such a low observed qprob.
        upper = qcount * kl_bounded(qprob, ema_prob)
        return upper >= SMA_Constants.binomial_thrsh

    #####

    # in dyal  (for dropping or discarding, trim_..,
    # trimming edges!)
    def prune_maps(self):
        self.updates_since_last_check = 0
        # tiny prob is defined as square of min_prob.
        tiny = self.min_prob * self.min_prob
        to_remove = Qs.compute_removables(
            self.c_to_q, self.max_map_entries, tiny)
        for c in to_remove:
            self.remove_item(c)

    # in dyal (prune or prunning, trim, etc)
    def remove_item(self, c):
        self.c_to_q.pop(c)
        if c in self.c_to_prob:
            self.c_to_prob.pop(c)
        if c in self.c_to_lr:
            self.c_to_lr.pop(c)
        if c in self.c_to_update:
            self.c_to_update.pop(c)
        if c in self.c_to_pos_update:
            self.c_to_pos_update.pop(c)
        assert len(self.c_to_q) >= len(self.c_to_prob), '0 position: %d %d' % ( len(self.c_to_q), len(self.c_to_prob) )

    # In per-edge (for trimming or prunning) ..
    def set_prune_schedule(self, thrsh):
        self.prune_schedule = thrsh

    ####

    # in dyal
    # 
    # (allocate if queue is not there and create=1)
    def get_queue(self, obs, create=1):
        q = self.c_to_q.get(obs, None)
        if q is None:
            if create: # create an empty queue, if not there?
                # self.c_to_q[obs] = ObsQ(self.q_capacity, self.q_smooth)
                q = ObsQ(self.q_capacity)# , self.q_smooth)
                self.c_to_q[obs] = q
        return q

    # in per-edge dyal: written specifically for simplified version
    # of update() below.
    def update_queues(self, obs):
        for c, q in self.c_to_q.items():
            self.c_to_update[c] += 1
            if c == obs:
                q.positive_update() # a positive update of queue q.
            else:
                q.increment_count()
    
    ###
    # in dyal
    #
    # Note: obs could be None too, which basically means weaken all.
    # 
    def update(self, obs, t=None):
        self.update_update_counts_etc(obs)
        # Should allocate queue, if not allocated (see get_queue below).
        qprob = None
        if obs is not None:
            q = self.get_queue(obs, create=1)
            qprob, qcount, _ = q.get_qinfo()

        assert len(self.c_to_q) >= len(self.c_to_prob), '1st: %d %d item:%s  %s' % (
            len(self.c_to_q), len(self.c_to_prob), obs, type(self.c_to_prob) )
            
        self.update_queues(obs) # Now update all the queues.

        assert len(self.c_to_q) >= len(self.c_to_prob), '2nd: %d %d' % ( len(self.c_to_q), len(self.c_to_prob) )
        
        # Weaken existing EMA weights too! and we are done if no queue.
        available = self.weaken_ema_weights_simple(obs)
        
        if qprob is None or qprob <= 0.0:
            # Nothing else to do (a new item).
            return
        
        # First check whether we need to listen to the queue..
        ema_prob = self.c_to_prob.get(obs, 0.0) # Existing EMA prob.
        # switch to the queue estimate?
        use_queue_prob = self.whether_to_use_queue(
            obs, ema_prob, q, qprob, qcount, check_lower=False)

        # Dont do anything (when qprob is too low and ema_prob is 0.0)..
        if qprob < self.min_prob and self.drop_below_min_prob and ema_prob == 0.0:
            return

        assert len(self.c_to_q) >= len(self.c_to_prob), '12th: %d %d' % ( len(self.c_to_q), len(self.c_to_prob) )
        
        # for the target obs, the lr (rate) was not decayed. Weaken it
        # here.
        if available <= 0:
            self.c_to_lr[obs] = self.harmonic_decay(self.c_to_lr[obs])
            return
        
        # Boost the EMA (edge) weight to the observation.  First
        # compute the delta (how much to add).
        if use_queue_prob:
            self.reset_to_qbigger += 1
            self.reset_to_q += 1
            # The prob. mass need for this update.
            delta = min(qprob - ema_prob, available)
            self.c_to_lr[obs] = 1.0 / qcount # my new learning rate.
        else:
            lr = self.c_to_lr[obs]
            # Normal EMA update: compute delta
            new_prob = (1-lr) * ema_prob + lr
            delta = min(new_prob - ema_prob, available)
            # Now decay the lr
            self.c_to_lr[obs] = self.harmonic_decay(lr)
        self.c_to_prob[obs] = ema_prob + delta # new ema prob

        assert self.c_to_prob[obs] > 0
        assert len(self.c_to_q) >= len(self.c_to_prob), '11th: %d %d' % ( len(self.c_to_q), len(self.c_to_prob) )

    # Update the update counts
    def update_update_counts_etc(self, obs):
        self.update_count += 1
        self.updates_since_last_check += 1
        if self.updates_since_last_check > self.prune_schedule:
            self.prune_maps()

        # all-update count maps (for other items) are
        # incremented below, before returning.
        if obs is not None: # if obs is None, it's like weakening for all.
            self.c_to_pos_update[obs] += 1

    ##

    def print_distro(self, sortit=1):
        print('\n# DYAL: \n')
        # print_distro(self.c_to_prob)
        pairs = list(self.c_to_prob.items())
        if sortit:
            pairs.sort(key=lambda x: -x[1])
        for c, p in pairs:
            q = self.c_to_q[c]
            lr = self.c_to_lr[c]
            print('%s:%.3f lr:%.3f (qProb:%.3f, qCount:%d)'% (
                c, p, lr, q.get_prob(), q.get_count()))

    # Note: if you use the queue, the counts are smaller.
    # but, we are not currently explicity keeping count for
    # each c (item) (we could!).
    def get_edge_update_count(self, c, use_queue=0):
        if use_queue: # in case you wanted to see the counts..
            q = self.c_to_q.get(c, None)
            if q is None:
                return 0
            # Note: the count in the queue could be small..
            # So not recommended for binomial tail (may incorrectly
            # imply there is not enough support)..
            return q.get_count()
        else:
            return self.c_to_update.get(c, 0)
        #return self.c_to_pos_update.get(c, 0)
        # assert False, 'not implemented'

    # in DYAL
    # get learning rate for edge to item c. May return None.
    def get_lr(self, c, lr=None):
        return self.c_to_lr.get(c, lr)

    # Return stats: max, median, min, on the learning rates.
    def get_lr_stats(self):
        lrs = []
        for c, lr in self.c_to_lr.items():
            lrs.append(lr)
        if lrs == []:
            return None, None, None, None
        # return np.max(lrs), np.median(lrs), np.min(lrs), len(self.c_to_q)
        return np.max(lrs), np.median(lrs), np.min(lrs), len(lrs)

    # For information only: get the pair or tuple (item and its lr and
    # pr) that had highest lr or maximum learning rate.
    def get_max_rate_tup(self):
        lrs = []
        maxr, maxc, prob = None, None, None
        for c, lr in self.c_to_lr.items():
            if maxr is None or maxr < lr:
                maxr, maxc, prob = lr, c, self.c_to_prob.get(c)
        return maxr, maxc, prob # max lr, item, and its pr

    # (in dyal) Return stats containing misc fields, eg number of
    # learning rates (number of edges), max lr, etc..
    def get_stats(self):
        maxlr, medianlr, minlr, numlr = self.get_lr_stats()
        if 0:
            median_w, max_w, totw = self.get_ema_stats()
        else:
            NS_distro, _ = SMA_Utils.get_SD_capped_filtered(self.c_to_prob)
            median_w, max_w, totw = self.get_ema_stats(NS_distro)

        return {'name': 'ema+dyal', 'max_lr': maxlr,
                'med_prob': median_w, 
                'max_prob': max_w, 
                'ema_tot': totw,
                'num_lr':numlr,
                 'num_updates': self.update_count,
                 'qmap_size': len(self.c_to_q)
                }

    def get_update_count(self):
        return self.update_count
    
    # (in dyal) Get probability stats (in ema map).
    # For now median, max, and sum total of probs.
    def get_ema_stats(self, distro=None):
        if distro is None:
            distro = self.c_to_prob
        ws = list(distro.values())
        if ws == []:
            return 0, 0, 0
        return np.median(ws), np.max(ws), np.sum(ws)        

    # How often DYAl switched to queues.
    def get_reset_counts(self):
        return self.reset_to_q, self.reset_to_qbigger, self.reset_to_qless

        
######


# Sparse EMA (multiclass EMA , SEMA) ..  Two variants are supported:
# static or plain fixed-rate SEMA (ie no change in the learning rate),
# and harmonic-decay SEMA (start the rate high and gradually lower it,
# hamonically) down to a minimum.
#
# >class >SEMA >static sparse >EMA (harmonic decay, when use_harmonic
# is set to true).
class EMA:
    def __init__(self, use_harmonic=False, min_rate=None,
                 max_rate=None, min_prob=None):
        self.min_prob = min_prob
        if self.min_prob is None:
            self.min_prob = SMA_Constants.min_prob

        # SMAs handle probabilities down to some minimum positive
        # probability.
        assert self.min_prob > 0

        self.use_harmonic = use_harmonic # use harmonic decay?
        self.reset() # create and initialize all maps
        if use_harmonic:
            self.rate = max_rate # initially set to highest allowed.
            if self.rate is None: # if maximum (starting) rate is not specified.
                self.rate = SMA_Constants.max_harmonic_lr # set rate (self.lr)
            # now, set the min_rate (when use_harmonic is true)
            self.min_rate = min_prob / 10.0
            if SMA_Constants.min_lr is not None:
                self.min_rate = SMA_Constants.min_lr
            if min_rate is not None:
                self.min_rate = min_rate
        else:
            # static: Don't have it too high.. as it won't change.
            self.rate = SMA_Constants.min_prob / 10.0
            if SMA_Constants.min_lr is not None:
                self.rate = SMA_Constants.min_lr
            if min_rate is not None:
                self.rate = min_rate

        # If a map size is below this, don't drop/prune (for EMA, DYAL and
        # Qs).  (some multiple of minimum probability supported, eg 1.5x)
        self.max_map_entries = int(1.5 / self.min_prob)
        # Check the map size (for pruning) every so often.
        self.prune_schedule = 1.5 * self.max_map_entries
        self.updates_since_last_check = 0                 
        # For informational purposes
        self.update_count = 0

    def reset(self):
        # item to (EMA) probability map (updated via EMA),
        # this is a semidistro.
        self.ema_sd = {}
        
    def get_prob(self, obs):
        return self.ema_sd.get(obs, 0.0)

    def predict_and_update(self, obs):
        prob = self.get_prob(obs)
        self.update(obs)
        return prob

    # in SEMA
    def update(self, obs):
        r = self.get_rate_etc(update=self.use_harmonic)
        plain_ema_update(obs, self.ema_sd, r)
        self.update_count += 1
        self.updates_since_last_check += 1                 
        if self.updates_since_last_check >= self.prune_schedule:
            self.prune_map()

    def get_name(self):
        if self.use_harmonic:
            return ("Sparse EMA using harmonic-decay, " +
                    "max or current rate=%.4f min-rate=%.4f") % (
                        self.rate, self.min_rate)
        else:
            return ("Sparse EMA, static fixed rate=%.4f") % \
                (self.rate)

    # (in static EMA) Get probability stats (in ema map).
    # For now median, max, and sum total of probs.
    def get_ema_stats(self):
        ws = list(self.ema_sd.values())
        if ws == []:
            return 0, 0, 0
        return np.median(ws), np.max(ws), np.sum(ws)

    # in sparse EMA
    def get_stats(self):
        median_w, max_w, totw = self.get_ema_stats()
        return {'name': 'SEMA',
                'med_prob': median_w, 
                'max_prob': max_w, 
                'ema_tot': totw,
                'num_updates': self.update_count,
                'num_weights': len(self.ema_sd),
                'lr': self.rate, 'min_lr': self.min_rate,
                'use_harmonic': self.use_harmonic
                }

    # posslibly updates the lr.
    def get_rate_etc(self, update=0):
        if not self.use_harmonic:
            # static EMA (the learning rate doesn't change).
            return self.rate
        else:
            rate = self.rate
            if update: # do a harmonic decay of rate.
                self.rate = harmonic_decay(self.rate, self.min_rate)
            return rate

    def get_distro(self):
        return self.ema_sd
    
    def get_lr(self):
        return self.rate

    # In SEMA
    def prune_map(self):
        self.updates_since_last_check = 0 # reset
        # we assume min_prob is sufficiently low, eg below 0.1, and
        # keep the highest 1.5 multiple of 1/min_prob.
        tiny = self.min_prob * self.min_prob
        to_remove = Qs.lowest_prs(
            self.ema_sd, self.max_map_entries, tiny)
        for c in to_remove:
            self.ema_sd.pop(c)
