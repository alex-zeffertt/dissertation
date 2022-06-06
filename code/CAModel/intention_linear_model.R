# Libraries needed
library(car)
library(arm)

# Read in the dataset produced by create_dataset2.py
# (which runs model_complete.py) 1000s of times
df = read.csv('dataset2.csv')
head(df)
dim(df)

# This shows 2963 "observations" were made.
#
# The dependent variable we're interested in is percent_intention.
# This is the percentage of the population that is in the "intending to reduce"
# stage of change after the CA has converged to a limit.
#
# Plot percent_intention against each independent variable to look for obvious
# relationships

# Jitter the x axis for plots where we only have a discrete levels in the data
jitter <- function(x) {
    sd <- 0.25 * min(diff(sort(unique(x))))
    return (x + rnorm(length(x),0,sd))
}

png(file='intention_linear_model1.png')
#dev.new()
par(mfrow=c(3,2))
plot(percent_intention ~ jitter(mean_n_weak_ties), data=df)
plot(percent_intention ~ jitter(modal_weak_tie_km), data=df)
plot(percent_intention ~ awareness_pc, data=df)
plot(percent_intention ~ facility_pc, data=df)
plot(percent_intention ~ p_update_logit_normal_sigma, data=df)
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

png(file='intention_linear_model2.png')
#dev.new()
par(mfrow=c(3,2))
plot(logit_intention ~ jitter(mean_n_weak_ties), data=df)
plot(logit_intention ~ jitter(modal_weak_tie_km), data=df)
plot(logit_intention ~ logit_awareness,data=df)
plot(logit_intention ~ logit_facility,data=df)
plot(logit_intention ~ p_update_logit_normal_sigma,data=df)
hist(df$logit_intention, xlab='logit_intention', main='Intention')
par(mfrow=c(1,1))
dev.off()

# That looks more promising.
# This is similar to with "percent_reducer" but the correlation with
# logit(facility_pc/100) is negative (because higher facility means it's
# easier to transition from "intention" to "reducer"
# Note that dependent var logit_intention is more or less normally distributed.

# Use scaled independent variables so that we can see the relative effect of
# each variable.

df$sc_mean_n_weak_ties  <- scale(df$mean_n_weak_ties)
df$sc_modal_weak_tie_km <- scale(df$modal_weak_tie_km)
df$sc_logit_awareness   <- scale(df$logit_awareness)
df$sc_logit_facility    <- scale(df$logit_facility)
df$sc_p_update_logit_normal_sigma <- scale(df$p_update_logit_normal_sigma)

# Create a linear model for logit_intention.
# Since there is an obvious curve in the correlation with logit_awareness
# and logit_facility we will use degree 2 polynomials for these
df$sc_logit_awareness_sq   <- scale(df$logit_awareness^2)
df$sc_logit_facility_sq    <- scale(df$logit_facility^2)

mod <- lm(logit_intention ~
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

# Drop sc_modal_weak_tie_km (NB: this is borderline)
mod <- lm(logit_intention ~
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
#                                Estimate Std. Error  t value Pr(>|t|)    
#(Intercept)                    -0.793588   0.004807 -165.079  < 2e-16 ***
#sc_mean_n_weak_ties             0.055426   0.004810   11.522  < 2e-16 ***
#sc_logit_awareness              0.493058   0.004814  102.429  < 2e-16 ***
#sc_logit_awareness_sq          -0.136184   0.004814  -28.288  < 2e-16 ***
#sc_logit_facility              -0.615777   0.004809 -128.041  < 2e-16 ***
#sc_logit_facility_sq           -0.161646   0.004810  -33.605  < 2e-16 ***
#sc_p_update_logit_normal_sigma -0.025624   0.004810   -5.327 1.07e-07 ***
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 0.2617 on 2956 degrees of freedom
#Multiple R-squared:  0.9083,	Adjusted R-squared:  0.9081 
#F-statistic:  4882 on 6 and 2956 DF,  p-value: < 2.2e-16


# 90.8% of the variation in the dependent variable "logit_intention" can be
# explained by this model.  All the remaining variables have very high
# confidence of correlation.  The most important is logit_facility, which
# accounts for a slope of -0.62 logit_reducer per sd (but this slope is reduced
# at higher values of logit_facility by the square term).  The interpretation
# of this is obvious: the easier society makes it to go from intending
# to reducing the the fewer intenders there will end up being and this has the
# greatest impact.  The next is logit_awareness, which contributes to
# logit_reducer with a positive slope of 0.49.  Clearly the more aware a
# population is (measured by the extent to which people with no connection
# to reducers are intending to reduce) the more intenders you will end up with.
# Far less important is the distribution of our resistance to change metric.
# However, if this is large this will result in a rump that would take a lot
# longer to transition between stages of change, and this is likely to cause
# a higher degree of relapse amoungst others.  The slope is -0.025 per
# standard deviation and in the simulation we used a uniform distribution between
# 0.25 (in which everyone is more or less equally resistant to change) and 3
# (in which the split is more bimodal between a population which will consider
# changing every time step and one which will barely ever consider changing).
# Just above this is the average number of weak ties
# simulated.  One sd in this contributed 0.055 to the logit_reducer.
# We tried values between 2 and 9 (remember these are social ties with whom
# people eat regularly but do not cohabit).  This was a bit of an unknown
# because the original H&L survey did not define this term very well.
# But luckily it seems the results of the model are fairly robust whatever we
# choose so it seems it doesn't matter too much.

# Repeat without the scale to get a formula for the model
summary(lm(logit_intention ~
               mean_n_weak_ties +
               logit_awareness +
               I(logit_awareness^2) +
               logit_facility +
               I(logit_facility^2) +
               p_update_logit_normal_sigma,
           data=df))


#Formula is therefore
#$$
# i = 100 logit^{-1}(
#   -0.56 logit(f) - 0.12 logit(f)^2 +
#    0.44 logit(a) - 0.10 logit(a)^2 +
#   -0.032 s +
#    0.027 m)
#$$
#where
#$$
# logit(x) = log\(\frac{x}{1-x}\)
#$$
#and
#
#| short | full | explanation |
#| i | percent_intention | % of population in intending to reduce stage, after model convergence |
#| f | logit_faciltiy | logit(proportion of reducers in population with no links to reducers excluding those with no intention to reduce). This is intended to represent how easy society as a whole makes meat reduction |
#| a | logit_awareness | logit(proportion with at least an intention to reduce in population with no links to reducers).  This is intended to represent how aware society is as a whole through public awareness campaigns etc. |
#| s | p_update_logit_normal_sigma | sd(logit(p_update)).  p_update is the probability an individual has of considering an update to its stage of change in one time step.  logit(p_update) is normally distributed with mean 0.5 and this value is its s.d..  A high value is interpreted as saying there is a great variability in resistance to change in the population |
#| m | mean_n_weak_ties | weak ties are ties which are not strong ties.  Although H&L is unclear about the exact definition we are taking it to mean people with whom an individual dines regularly other than cohabitants.  This variable is the mean of that distribution, which is taken to be a poisson distribution. |

# Create plot to show the reliability of the model and check assumptions (e.g.
# normally distributed residuals)
df$predicted <- predict(mod, df)
png(file='intention_linear_model3.png')
#dev.new()
par(mfrow=c(2,2))
plot(mod, which=1)
plot(mod, which=2)
plot(logit_intention ~ predicted, data=df, main='Reliability of linear model')
hist(residuals(mod))
par(mfrow=c(1,1))
dev.off()

# This all looks good.  residuals normally distributed.  QvsQ chart shows normal
# over -3 to +3 quantiles range.  Some bunching of residuals vs preducted value,
# suggesting that a non linear model may be better. However, the third plot shows
# that even a linear model is still a very good predictor.

max(cooks.distance(mod))
# This is <<1
