# Libraries needed
library(car)
library(arm)

# Read in the dataset produced by create_dataset2.py
# (which runs model_complete.py) 1000s of times
# using variable beta values (as per the uncertainty in their estimates)
df = read.csv('dataset3.csv')
head(df)
dim(df)

# This shows 2963 "observations" were made.
#
# The dependent variable we're interested in is percent_reducer.
# This is the percentage of the population that is in the "meat reducer"
# stage of change after the CA has converged to a limit.
#
# Plot percent_reducer against each independent variable to look for obvious
# relationships

# Jitter the x axis for plots where we only have a discrete levels in the data
jitter <- function(x) {
    sd <- 0.3 * min(diff(sort(unique(x))))
    return (x + rnorm(length(x),0,sd))
}

png(file='sensitivity_linear_model1.png')
#dev.new()
par(mfrow=c(3,2))
plot(percent_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(percent_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(percent_reducer ~ awareness_pc, data=df)
plot(percent_reducer ~ facility_pc, data=df)
plot(percent_reducer ~ p_update_logit_normal_sigma, data=df)
par(mfrow=c(1,1))
dev.off()

# There is some obvious relation with awareness_pc and facility_pc.
# However, it doesn't look particularly linear.  Noting that percentages
# are bounded at 0 and 100 it makes sense to replace these with logit values,
# i.e. log ratios

# Create alternative variables for the logit transformed percentages
df$logit_reducer   <- logit(df$percent_reducer/100)
df$logit_intention <- logit(df$percent_intention/100)
df$logit_awareness <- logit(df$awareness_pc/100)
df$logit_facility  <- logit(df$facility_pc/100)

png(file='sensitivity_linear_model2.png')
#dev.new()
par(mfrow=c(3,2))
plot(logit_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(logit_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(logit_reducer ~ logit_awareness,data=df)
plot(logit_reducer ~ logit_facility,data=df)
plot(logit_reducer ~ p_update_logit_normal_sigma,data=df)
hist(df$logit_reducer, xlab='logit_reducer', main='Reducers')
par(mfrow=c(1,1))
dev.off()

# That looks more promising
# Note that dependent var logit_intention is more or less normally distributed.

# Use scaled independent variables so that we can see the relative effect of
# each variable.

df$sc_mean_n_weak_ties  <- scale(df$mean_n_weak_ties)
df$sc_modal_weak_tie_km <- scale(df$modal_weak_tie_km)
df$sc_logit_awareness   <- scale(df$logit_awareness)
df$sc_logit_facility    <- scale(df$logit_facility)
df$sc_p_update_logit_normal_sigma <- scale(df$p_update_logit_normal_sigma)

# Create a linear model for logit_reducer.
# Since there is an obvious curve in the correlation with logit_awareness
# and logit_facility we will use degree 2 polynomials for these
df$sc_logit_awareness_sq   <- scale(df$logit_awareness^2)
df$sc_logit_facility_sq    <- scale(df$logit_facility^2)

mod <- lm(logit_reducer ~
              sc_mean_n_weak_ties +
              sc_modal_weak_tie_km +
              sc_logit_awareness +
              sc_logit_awareness_sq +
              sc_logit_facility +
              sc_logit_facility_sq +
              sc_p_update_logit_normal_sigma,
          data=df)

# Look for a more parsimonious model.  We will use the method of
# removing the independent variable that leaves AIC the lowest,
# as long as the overall AIC does not increase by more than 2.

# Run dropterm
dropterm(mod,test='F')

# Drop sc_modal_weak_tie_km
mod <- lm(logit_reducer ~
              sc_mean_n_weak_ties +
              sc_logit_awareness +
              sc_logit_awareness_sq +
              sc_logit_facility +
              sc_logit_facility_sq +
              sc_p_update_logit_normal_sigma,
          data=df)

# Re-run dropterm
dropterm(mod,test='F')

# No further independent variables can be removed
# Let's have a look at the result
summary(mod)

#Coefficients:
#                                Estimate Std. Error t value Pr(>|t|)    
#(Intercept)                    -0.615426   0.008410 -73.178  < 2e-16 ***
#sc_mean_n_weak_ties             0.034455   0.008414   4.095 4.29e-05 ***
#sc_logit_awareness              0.510149   0.008415  60.624  < 2e-16 ***
#sc_logit_awareness_sq          -0.160519   0.008414 -19.078  < 2e-16 ***
#sc_logit_facility               0.937771   0.008413 111.469  < 2e-16 ***
#sc_logit_facility_sq           -0.122884   0.008416 -14.602  < 2e-16 ***
#sc_p_update_logit_normal_sigma -0.079876   0.008418  -9.488  < 2e-16 ***
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 0.5747 on 4662 degrees of freedom
#Multiple R-squared:  0.7805,	Adjusted R-squared:  0.7802 
#F-statistic:  2763 on 6 and 4662 DF,  p-value: < 2.2e-16

# 78% of the variation in the dependent variable "logit_reducer" can be
# explained by this model.  All the remaining variables have very high
# confidence of correlation.  The most important is logit_facility, which
# accounts for a slope of +0.94 logit_reducer per sd (but this slope is reduced
# at higher values of logit_facility by the square term).  The interpretation
# of this is obvious: the easier society makes it to go from intending
# to reducing the the more reducers there will end up being and this has the
# greatest impact.  The next is logit_awareness, which contributes to
# logit_reducer with a positive slope of 0.51.  Clearly the more aware a
# population is (measured by the extent to which people with no connection
# to reducers are intending to reduce) the more reducers you will end up with.
# Far less important is the distribution of our resistance to change metric.
# However, if this is large this will result in a rump that would take a lot
# longer to transition between stages of change, and this is likely to cause
# a higher degree of relapse amoungst others.  The slope is -0.080 per
# standard deviation and in the simulation we used a uniform distribution between
# 0.25 (in which everyone is more or less equally resistant to change) and 3
# (in which the split is more bimodal between a population which will consider
# changing every time step and one which will barely ever consider changing).
# The least important explanatory variable was the average number of weak ties
# simulated.  One sd in this only contributed 0.034 to the logit_reducer.
# We tried values between 2 and 9 (remember these are social ties with whom
# people eat regularly but do not cohabit).  This was a bit of an unknown
# because the original H&L survey did not define this term very well.
# But luckily it seems the results of the model are fairly robust whatever we
# choose so it seems it doesn't matter too much.

# Repeat without the scale to get a formula for the model
summary(lm(logit_reducer ~
               mean_n_weak_ties +
               logit_awareness +
               I(logit_awareness^2) +
               logit_facility +
               I(logit_facility^2) +
               p_update_logit_normal_sigma,
           data=df))


#Formula is therefore
#$$
# r = 100 logit^{-1}(
#    0.94 logit(f) - 0.12 logit(f)^2 +
#    0.51 logit(a) - 0.16 logit(a)^2 +
#   -0.080 s +
#    0.034 m)
#$$
#where
#$$
# logit(x) = log\(\frac{x}{1-x}\)
#$$
#and
#
#| short | full | explanation |
#| r | percent_reducer | % of population in meat reducer stage after model convergence |
#| f | logit_faciltiy | logit(proportion of reducers in population with no links to reducers excluding those with no intention to reduce). This is intended to represent how easy society as a whole makes meat reduction |
#| a | logit_awareness | logit(proportion with at least an intention to reduce in population with no links to reducers).  This is intended to represent how aware society is as a whole through public awareness campaigns etc. |
#| s | p_update_logit_normal_sigma | sd(logit(p_update)).  p_update is the probability an individual has of considering an update to its stage of change in one time step.  logit(p_update) is normally distributed with mean 0.5 and this value is its s.d..  A high value is interpreted as saying there is a great variability in resistance to change in the population |
#| m | mean_n_weak_ties | weak ties are ties which are not strong ties.  Although H&L is unclear about the exact definition we are taking it to mean people with whom an individual dines regularly other than cohabitants.  This variable is the mean of that distribution, which is taken to be a poisson distribution. |

# Create plot to show the reliability of the model and check assumptions (e.g.
# normally distributed residuals)
df$predicted <- predict(mod, df)
png(file='sensitivity_linear_model3.png')
#dev.new()
par(mfrow=c(2,2))
plot(mod, which=1)
plot(mod, which=2)
plot(logit_reducer ~ predicted, data=df, main='Reliability of linear model')
hist(residuals(mod))
par(mfrow=c(1,1))
dev.off()

max(cooks.distance(mod))
# This is well below 1 so no problem

# This all looks good.  residuals normally distributed.  QvsQ chart shows normal
# over -2 to +2 sd range.  Some bunching of residuals vs preducted value, but not
# between the -1 to +1 range of logit_reducer (which equates to between 25% and 75%
# of the population being reducers)
