# Each person will have a different probability of considering changing their
# diet in each time step because resistance to change differs from person to
# person.  If we know what that probability is for a given individual then we
# can decide at each time step whether to leave their state the same, or
# update their state randomly.  If we update their state randomly we use
# the probabilities based on their social ties to randomly select a new state.

# Let's call the a person's probability of considering a change in one time step, or x.
# In Generalized Linear Models where a response variable x is a probability or a proportion,
# a link function y = logit(x) is used, and the residuals of y, not x, are
# are normally distributed.  In this case there are no independent variables so y itself is
# normally distrubuted.  The link function transforms from a bounded domain [0,1] to an unbounded
# domain [-\infty, +\infty] which can be used with a normal distribution.
#
# We want to find a distribution for x, also known as a PDF or dp/dx.
# We can assume that the mean is 0.5 if we do not fix the time step.  (We have nominally
# declared a timestep is 6 months, but nothing depends on this.)
# We cannot use the normal distribution N(0.5, sigma) because the domain of x is
# restricted to [0,1] not [-\infty, +\infty].
# If we define y = logit(x) that transforms an x-domain of [0,1] to a y-domain of  [-\infty, +\infty]
# so let's assume that y is normally distributed and see what distributions this gives us

# It turns out it gives the logit normal distribution
#
#    P(N(\mu, \sigma))
#
# with pdf
#
#    \frac{1}{\sigma{\sqrt{2\pi }}}}\,{\frac{1}{x(1-x)}}\,e^{{-{\frac{(\operatorname {logit}(x)-\mu )^{2}}{2\sigma^{2}}}}
#
# It's a distribution where the logit transformed variable y = logit(x)
# normally distributed by the normal distribution.

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import scipy

# Define the domain in terms of x and of its transform y
epsilon = 0.0001
x = np.linspace(0+epsilon,1-epsilon, 1000)
y = np.log(x/(1-x)) # equiv to scipy.special.logit(x)
dydx = (1/x + 1/(1-x))

# Function which returns dp/dx given a standard deviation of dp/dy
def pdf(sigma):
    dpdy = norm.pdf(y,0,sigma)
    dpdx = dpdy * dydx
    return dpdx

# Plot pdf dp/dx for several values of sigma
fig = plt.figure()
for sigma in np.arange(.25,2,.25):
    plt.plot(x, pdf(sigma), label=f'$\sigma={sigma}$')

plt.legend()
plt.ylabel('$\\frac{d\.Pr(p)}{dp}$', rotation=0)
plt.xlabel("p = individual's probability of considering change in one time step")
plt.title('Distribution of $p$ given\n$\\frac{d\.Pr(x)}{dx}$ = N(0,$\sigma$)\nx = logit(p)')
fig.subplots_adjust(top=fig.subplotpars.top * .95)
plt.ion()
plt.show()

def cdf(sigma):
    csum = np.cumsum(pdf(sigma))
    return csum/csum[-1]

# Plot cdf p(<x) for several values of sigma
fig = plt.figure()
for sigma in np.arange(.25,2,.25):
    plt.plot(x, cdf(sigma), label=f'$\sigma={sigma}$')

plt.legend()
plt.ylabel('p', rotation=0)
plt.xlabel("x = individual's probability of considering change in one time step")
plt.title('Cummulative distribution of prob(change_considered_one_timestep) using\n$\\frac{dp}{dy}$ = N(0,$\sigma$)\ny = logit(x)')
fig.subplots_adjust(top=fig.subplotpars.top * .95)
plt.ion()
plt.show()
# NB:
# at sigma = 0.25 the probabilities are tightly distributed around 1/2
# by sigma = 1.5 it has flattened to a more or less uniform distribution between
#            zero and 1
# after sigma = 1.75 the distribution starts to become bimodal with one mode near
#            zero and another mode close to 1
