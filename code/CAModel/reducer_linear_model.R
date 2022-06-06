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
# The dependent variable we're interested in is percent_reducer.
# This is the percentage of the population that is in the "meat reducer"
# stage of change after the CA has converged to a limit.
#
# Plot percent_reducer against each independent variable to look for obvious
# relationships

# Jitter the x axis for plots where we only have a discrete levels in the data
jitter <- function(x) {
    sd <- 0.25 * min(diff(sort(unique(x))))
    return (x + rnorm(length(x),0,sd))
}

png(file='reducer_linear_model1.png')
#dev.new()
par(mfrow=c(3,2))
plot(percent_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(percent_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(percent_reducer ~ awareness_pc, data=df)
plot(percent_reducer ~ facility_pc, data=df)
plot(percent_reducer ~ p_update_logit_normal_sigma, data=df)
hist(df$percent_reducer, xlab='percent_reducer', main='Reducers')
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

png(file='reducer_linear_model2.png')
#dev.new()
par(mfrow=c(3,2))
plot(logit_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(logit_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(logit_reducer ~ logit_awareness,data=df)
plot(logit_reducer ~ logit_facility,data=df)
plot(logit_reducer ~ p_update_logit_normal_sigma,data=df)
hist(logit(df$percent_reducer/100), xlab='logit_reducer', main='Reducers')
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


# Repeat without the scale to get a formula for the model
summary(lm(logit_reducer ~
               mean_n_weak_ties +
               logit_awareness +
               I(logit_awareness^2) +
               logit_facility +
               I(logit_facility^2) +
               p_update_logit_normal_sigma,
           data=df))

# Create plot to show the reliability of the model and check assumptions (e.g.
# normally distributed residuals)
df$predicted <- predict(mod, df)
png(file='reducer_linear_model3.png')
#dev.new()
par(mfrow=c(2,2))
plot(mod, which=1)
plot(mod, which=2)
plot(logit_reducer ~ predicted, data=df, main='Reliability of linear model')
hist(residuals(mod))
par(mfrow=c(1,1))
dev.off()

max(cooks.distance(mod))

# This all looks good.  residuals normally distributed.  QvsQ chart shows normal
# over -2 to +2 sd range.  No bunching of residuals vs preducted value.
