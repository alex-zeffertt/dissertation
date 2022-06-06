# Libraries needed
library(car)
library(arm)

# Read in the dataset produced by create_dataset4.py
# (which runs model_complete.py) 1000s of times
df = read.csv('dataset4.csv')
head(df)
dim(df)

jitter <- function(x) {
    sd <- 0.25 * min(diff(sort(unique(x))))
    return (x + rnorm(length(x),0,sd))
}

#png(file='reducer_feedback_linear_model1.png')
dev.new()
par(mfrow=c(3,2))
plot(percent_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(percent_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(percent_reducer ~ awareness_pc, data=df)
plot(percent_reducer ~ facility_pc, data=df)
plot(percent_reducer ~ p_update_logit_normal_sigma, data=df)
plot(percent_reducer ~ feedback, data=df)
par(mfrow=c(1,1))
#dev.off()

#png(file='reducer_feedback_linear_model1.5.png')
dev.new()
hist(df$percent_reducer, xlab='percent_reducer', main='Reducers')
#dev.off()

# There is some obvious relation with awareness_pc and facility_pc.
# However, it doesn't look particularly linear.  Noting that percentages
# are bounded at 0 and 100 it makes sense to replace these with logit values,
# i.e. log ratios.  Also replace feedback as this is bounded from 0 to 1

# Create alternative variables for the logit transformed percentages
df$logit_reducer   <- logit(df$percent_reducer/100)
df$logit_intention <- logit(df$percent_intention/100)
df$logit_awareness <- logit(df$awareness_pc/100)
df$logit_facility  <- logit(df$facility_pc/100)
df$logit_feedback  <- logit(df$feedback)

#png(file='reducer_feedback_linear_model2.png')
dev.new()
par(mfrow=c(3,2))
plot(logit_reducer ~ jitter(mean_n_weak_ties), data=df)
plot(logit_reducer ~ jitter(modal_weak_tie_km), data=df)
plot(logit_reducer ~ logit_awareness,data=df)
plot(logit_reducer ~ logit_facility,data=df)
plot(logit_reducer ~ p_update_logit_normal_sigma,data=df)
plot(logit_reducer ~ logit_feedback,data=df)
par(mfrow=c(1,1))
#dev.off()

#png(file='reducer_feedback_linear_model2.5.png')
dev.new()
hist(df$logit_reducer, xlab='logit_reducer', main='Reducers')
#dev.off()

# initial model
mod <- lm(logit_reducer ~
              mean_n_weak_ties +
              modal_weak_tie_km +
              logit_awareness +
              I(logit_awareness^2) +
              logit_facility +
              I(logit_facility^2) +
              p_update_logit_normal_sigma +
              logit_feedback
        , data=df)

dropterm(mod,test='F')

# Drop mean_n_weak_ties
mod <- lm(logit_reducer ~
              modal_weak_tie_km +
              logit_awareness +
              I(logit_awareness^2) +
              logit_facility +
              I(logit_facility^2) +
              p_update_logit_normal_sigma +
              logit_feedback
        , data=df)

dropterm(mod,test='F')

# Drop modal_weak_tie_km
mod <- lm(logit_reducer ~
              logit_awareness +
              I(logit_awareness^2) +
              logit_facility +
              I(logit_facility^2) +
              p_update_logit_normal_sigma +
              logit_feedback
        , data=df)

dropterm(mod,test='F')

# No more terms can be dropped without increasing AIC by more than 2
summary(mod)

# 93.1% variation explained

summary(lm(logit_reducer ~
              scale(logit_awareness) +
              I(scale(logit_awareness)^2) +
              scale(logit_facility) +
              I(scale(logit_facility)^2) +
              scale(p_update_logit_normal_sigma) +
              scale(logit_feedback)
        , data=df))

#Coefficients:
#                                    Estimate Std. Error t value Pr(>|t|)    
#(Intercept)                         0.660524   0.016665   39.63   <2e-16 ***
#scale(logit_awareness)              0.422682   0.009990   42.31   <2e-16 ***
#I(scale(logit_awareness)^2)        -0.114801   0.009341  -12.29   <2e-16 ***
#scale(logit_facility)               1.183098   0.009977  118.58   <2e-16 ***
#I(scale(logit_facility)^2)         -0.150659   0.009293  -16.21   <2e-16 ***
#scale(p_update_logit_normal_sigma) -0.111596   0.009972  -11.19   <2e-16 ***
#scale(logit_feedback)               0.882118   0.009979   88.40   <2e-16 ***
#---
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#Residual standard error: 0.4285 on 1842 degrees of freedom

# Feedback term appears similarly relevant as logit_facility, when these are scaled, and ~ twice as important as awareness.
# Other terms relatively unaltered.  NB: feedback range was 0 to 1
