#
import random, time
#import math, numpy as np

from math import log

from collections import defaultdict, Counter

##

# timer!
def get_mins_from( from_time ):
    delta = time.time() - from_time
    return delta / 60.0

def get_hrs_from( from_time ):
    delta = time.time() - from_time
    return delta / 3600.0

##

# KL(p || q), where both are maps..
# divergence of distro q from p.
#
def kl_general(ps, qs):
    s = 0
    for i, p in ps.items():
        q = qs.get(i, 0.0)
        assert q > 0 # what to do with 0s?
        s += p * log(p / q)
    return s
##

# bounded kl() or relative_entropy() ..  for the binary case..
def kl_bounded(a, b):
    """a and b are probabilities, and their order matters.  So
    relatively entropy of b with respect to a (or divergence of b from
    a)..  Using natural log or ln (for 'nats').

    """
    assert(b > 0)
    if a >= 1.0:
        return a * log(a / b)  # short circuit.  
    if b >= 0.999: # so 1-b is not zero..
        b = 0.999
    return a * log(a / b) + (1 - a) * log(  (1-a) / (1-b) )

####### Losses

# In these, we assume probs1 and probs2 are distributions or at least
# semidistributions..

# sample according to probs1, but incur costs accoring to probs2. This
# is the logloss scoring rule.
def expected_logloss(probs1, probs2):
    s = 0.0
    for c, p in probs1.items():
        s += p * -log(probs2.get(c))
    return s

# This turned out to not be proper!!
def expected_quadloss(probs1, probs2):
    s, sump = 0.0, 0.0
    for c, p in probs1.items():
        d = (1.0 - probs2.get(c)) * (1.0 - probs2.get(c)) * p
        s +=  d
        print(d, '  sum=', s)
        sump += p
    assert sump == 1.0
    return math.sqrt(s)

def brier_score(preds_list, obs, k=2):
    if type(preds_list) == dict:
        preds_list = list(preds_list.items())
    lk = 0
    found, sump = 0, 0.0
    for s2, p2 in preds_list: # Brier form of scoring
        sump += p2
        if s2 == obs:
            lk += pow(1.0 - p2, k)
            found = 1
        else:
            lk += pow(p2, k)
    if not found: # Make sure you punish
        lk += 1   # if not there (OOV)..
        # NOTE: if not there, and no other items predicted,
        # loss is 1.0, but otherwise, Brier score (loss) can be up to 2.0.
    assert sump <= 1.001, '# sump was %.5f' % sump
    return lk

# Brier score, or scoring, rule for multiclass, min is 0, max is 2.
# (Brier distances or diffs.. Brier distance is a better name than
# Brier score.. but i guess if you are assessing the calibration of a
# model.. a 'score' is a better term..).. k is the power, and k=2,
# is the standard Brier.
def expected_brier_score(probs1, probs2, k=2):
    s, sump = 0.0, 0.0
    for c1, p1 in probs1.items():
        d, found = 0, False
        # Brior reqiure you to go over all elements in p2.
        for c2, p2 in probs2.items():
            if c2 == c1:
                d += pow(abs(1.0 - p2), k) # * (1.0 - p2)
                found = 1
            else:
                d += pow(p2, k) # * p2
        if not found:
            d += 1
        s +=  d * p1
        #print(d, '  sum=', s)
        sump += p1
    assert sump <= 1.001, 'sump was %.3f' % sump
    return s


