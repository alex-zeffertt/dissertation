#!/usr/bin/env python3
#
# Experimentation with the results from H&L
# to try and reverse their published data
# and produce probabilities that can be used in CAModel.py

import numpy as np
import matplotlib.pyplot as plt

# Categorical explanatory variable X
# ==================================
# There are 5 categories for the number of ties an individual might have
#
#   description                                    j
# * 0 strong ties and 0 weak ties  (reference)     0
# * 0 strong ties and 1-2 weak ties                1
# * 1-2 strong ties, 0 weak ties                   2
# * 1-2 strong ties, >0 weak ties                  3
# * 3+ strong ties                                 4
#
# Thus x is a single 
N_SOCIAL = 5

#  Categorical response variable Y
#  ===============================
#
# There are 3 categories for the state of an individual
#
#   description                                    i
# * NO intention to reduce meat intake (reference) 0
# * Intention to reduce meat intake                1
# * Meat reducer                                   2

# The multinomial logistic model that was built is
#
#   P(Y = i | X = j) = P(Y = 0 | X = j) exp(C_i + beta_ij)
#
# for i = 1,2
#
# where C_i and beta_ij coefficients corresponding to the outcome i and case X = j.
# Since j = 0 is the reference case we know that
#
#    beta_i0 = 0
#
# We also know all the other beta_ij for i=1,2 because they are reported in Hielkema & Lund.
# We also know by substitution when i = j = 0 that C_0 = 0.
# Therefore, by substitution we get also that beta_0j = 0

# Best estimate for beta
beta = np.array([
    [0.0, 0.0,  0.0,  0.0,  0.0],
    [0.0, 0.69, 2.78, 3.81, 0.65],   # Intention vs NO intention
    [0.0, 0.73, 2.54, 3.19, 3.30]])  # Reducer   vs NO intention

# If we know C_1 and C_2 we can work out P(Y = i | X = j) since
# (from https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
#
#    P(Y = i | X = j) = exp(C_i + beta_ij) / ( 1 + sum_k=1,2 {exp(C_k + beta_kj)} )
#
# which, given C_0 = beta_0j = 0 this simplifies to
#
#    P(Y = i | X = j) = exp(C_i + beta_ij) / sum_k=0..2 {exp(C_k + beta_kj)}
#
# The above formula implies that if we are given that X = 0 we know that the ratio
# P(Y = 0) : P(Y = 1) : P(Y = 2) must have the form
#
#    1 : exp(C_1) : exp(C_2)
#
# Another way of putting it is that the constants C_1 and C_2 represent the predisposition, in
# the absence of meat-reducer social ties, to be either intending reduce to or actively reducing
# meat consumption, with
#
#    C1 = log(pop_intending_with_no_ties/pop_not_intending_with_no_ties)
#    C2 = log(pop_reducing_with_no_ties/pop_not_intending_with_no_ties)
#
# These are parameters which are culturally defined, but which are also changable, with C_1 being
# dependent on climate, health, and welfare awareness, and C_2 being dependent on both C_1 and
# on practicality of adopting a meat reduced diet.  Although meat-reducers (Y = 2) may technically
# include people who have never passed through the Y = 1 stage and have always maintained a meat free diet
# (e.g. for religious reasons or a sense of disgust towards meat) this is a small number in a country
# like Denmark (it must be less than the number of vegetarians/vegans i.e. 3.5%) and therefore can be
# ignored in a model like this.
#
# Therefore C_1 and C_2 are parameters we can play with to determine how awareness and practicality can
# affect the evolution of the model.


# Create plots for a number of different options for C_1 and C_2
plt.ion()
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
plt.suptitle('Effect of social network on meat-reduction Stage of Change')

percent_splits = [
    [ 90, 9, 1 ],
    [ 60, 25, 15],
    [ 80, 15, 5 ],
    [ 70, 20, 10 ],
]

for percent, ax in zip(percent_splits, axes.flatten()):
    C = np.log(np.array([percent])/percent[0])
    C = C.transpose()

    # This allows us to work out
    P = np.exp(C + beta)/np.exp(C + beta).sum(axis=0)

    # Plot the result
    plt.sca(ax)
    plt.title(f'No-ties split {percent[0]}:{percent[1]}:{percent[2]}')
    plt.bar(x = np.arange(5) +.00, height=P[0], width=.25, color='red',   edgecolor='black', alpha=.5, label='Pr(NO intention)')
    plt.bar(x = np.arange(5) +.25, height=P[1], width=.25, color='yellow',edgecolor='black', alpha=.5, label='Pr(Intention)')
    plt.bar(x = np.arange(5) +.50, height=P[2], width=.25, color='green', edgecolor='black', alpha=.5, label='Pr(Reducer)')


# Stuff shared between axes
plt.xticks(np.arange(5) +.25,('No ties','1-2 weak','1-2 strong','1-2 strong\n& weak', '3+ strong'))
axes[0][1].legend()
axes[1][0].set_xlabel('Number of meat-reducer social-ties')
axes[1][1].set_xlabel('Number of meat-reducer social-ties')
axes[0][0].set_ylabel('Probability')
axes[1][0].set_ylabel('Probability')

# NB: We know from the H&L paper that in the Danish population as a whole we have the following split
#
#  NO intention    Intention    Reducer
#  57.3%           11.4%        31.2%
#
# This generates an upper bound for a realistic C_1 and C_2, namely log(11.4/57.3) and log(31.2/57.3)
# Since the ratios must be smaller if we restrict to just the population that have no ties to meat-reducers.

# Let's transform C_1 and C_2 into two more easily characterized variables
# 1. Awareness
# 
# This is the percentage of the no-ties population that are
# either Intending to reduce or actively reducing meat consumption.  We'll call
# this the societal awareness because it can only depend on global factors such
# as climate campaigns, and not information from meat-reducer peers.
#
#    Awareness := 100 * (no_ties_intention + no_ties_reducer) / (no_ties_all)
#
# 2. Facility
#
# This is the percentage of the "aware" no-ties population that are active
# reducers.  This represents how easy society as a whole makes being a meat
# reducer, and includes availability of ingredients and catered options,
# skills and information, motivation
#
#    Facility := 100 * (no_ties_reducer) / (no_ties_intention + no_ties_reducer)


print(f"{'NO intent':>10s}% {'Intention':>10s}% {'Reducer':>10s}% {'awareness':>10s}%  {'facility':>10s}%")
for percent in percent_splits:
    awareness = percent[1] + percent[2]
    facility = 100 * percent[2] / awareness
    print(f"{percent[0]:10d}% {percent[1]:10d}% {percent[2]:10d}% {awareness:10.1f}%  {facility:10.1f}%")




