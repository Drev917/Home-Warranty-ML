#**********************************************************************************************************
# Home Warranty case study  - Logistic regression and evaluation of model                                 *
# This is a group case study                                                                              *
#                                                                                                         *
# Members:                                                                                                *
# 1.Drew Barton                                                                                           *     
# 2.Chris Calvo                                                                                           *
# 3.Jake Pagels                                                                                           *                                                                                      
#                                                                                                         *
# Brief on subject:                                                                                       * 
#---------------------------------------------------------------------------------------------------------*
# For a home warranty company, the rate at which customers renew after their contract expires is a        *
# crucial metric to the company's ability to cover acquisition costs and increase profitability.          *
# The more times a customer renews the longer the tenure of the customer increasing customer lifetime     *
# value and profits for the company.                                                                      *
#                                                                                                         *
# Goal: To create a model with the probability of renewal using a logistic regression.                    *
#                                                                                                         *
# Input file used:                                                                                        *
#                                                                                                         *
# 1. Home Warranty Success.xlsx - consist of a list of home warranties (12 month contracts), demographic  *
#                                 of customer, revenue received, claims filed, claims expense per         *
#                                 warranty, options purchases, contact information, tenure of customer,   *
#                                 address including state, renewal price, price step up (renewal price -  *
#                                 price paid), consumer sentiment index, housing trends and success of    *
#                                 renewal (1 - renewed, 0 - did not renew)                                *                                                                   
#**********************************************************************************************************

##Column clarification breakdown

# `Warranty key`: Customer identification
# `Brochure Name`: Type of coverage
# `Supreme in Base`: Whether the standard coverage had Supreme.  Binary format.  
# `Washer/Dryer in Base`: Whether the standard coverage had washer/dryer.  Binary format.
# `Septic in Base`: Whether the standard coverage had septic.  Binary Format.
# `House over 5,000 Sq Ft`: Houses over 5,000 ft get charged $150 to $300 more. Binary format
# `Condo`: Condos get charged $20 less. Binary format
# `High Risk`: Whether something about the home made it a high risk to insure so we charged it a $50 surcharge. Binary format. 
# Column K to L: Whether the customer provided a phone, email or created a web profile.  This is contact information and normally when they provide us more they renew better.
# `Sell Date`: When the warranty was sold
# `Expiration Date`: When the warranty expired.  We start marketing 90 days before expiration and start calling 2 weeks before expiration to renew them.
# `Primary Sales Channel`: How we sold the customer
# `Payment Method`: How often they pay us and if they paid with check or credit. Full means pay up front. Monthly pay us each month. Easy pay us the first 3 months for the full year
# `Auto Renew`: If they are set up to automatically renew so our sales team doesn't have to call
# `Base Price`:  How much their standard coverage costs
# `Option Price`: How much the extra coverage they purchase costs
# `Gross Price`:  Base + Options
# `Discount`: How much discount was offered to the Gross Revenue
# `Net Price`: Gross + Discount
# `Deductible`: How much they have to pay us before we can service their warranty
# `Claim Count`:  How many times they called for a claim
# `Had Claim`: If they had a claim then "Yes" otherwise "No"
# `Claim Expense`: How much their claims were worth
# `High Claims`: We have paid out so much claims on the customer over their tenure we charge them a $150 surchage.  Binary format
# Columns AG to AY: Each one of these columns is an option or extra coverage on top of the standard coverage that we sell for an extra price.  
# If the customer purchased the option it is marked as 1 otherwise 0.  The sum of their options are in the Option Purchased column.  
# Most important options are Supreme, Service Fee Buydown and Washer/Dryer, and the rest are subsetted out.
# `Options Purchased`: Number of options purchased.  Sum of columns AG to AY.
# `Purchased Option`: Yes/No. Binary format
# `Renewal Base Step Up`: How much is their base price increasing on renewal
# `Renewal Option Step Up`:  How much is their option price increasing on renewal
# `Renewal Gross Step Up`: Add the two above
# `Renewal Discount Step Up`: How much is their discount expected to increase. if -$120 we are giving them $120 more in discount dollars
# `Renewal Net Step Up`:  How much is their overall price increasing
# `Renewed`:  Most importantly did they end up renewing with us after their warranty expired.
# Columns BH to BL: NAHB Housing Market Index and lagged.  How well the housing market is doing
# Columns BM to BQ: Consumer Sentiment Index and lagged

#--------------------------------------------------------------------------------------------------------------

setwd("D:/Drewb/Desktop/R Files")

#libraries used in analysis after installing all packages

library(dplyr)
library(ggplot2)
library(ggthemes)
library(reshape2)
library(corrplot)
library(car)
library(readxl)
library(boot)
library(janitor)
library(boot)
library(MASS)
library(olsrr)
library(caTools)
library(caret)
library(e1071)
library(ROCR)
library(pROC)
library(broom)
library(modelr)
library(rpart)
library(rpart.plot)
library(randomForest)


################################################################################################################
#EDA and Data Wrangling
################################################################################################################

home_warranty_base = data.frame(read_excel("Home Warranty Success.xlsx"))
dim(home_warranty_base)
colnames(home_warranty_base)

#checking for duplicates in key field
sum(duplicated(home_warranty_base$Warranty.key)) # 0

#checking for NA's in df
sum(is.na(home_warranty_base)) # 16

#determining which fields have NA's
sapply(home_warranty_base, function(x) length(which(is.na(x))))

#8 NA's in `Renewal.Base.Step.Up` and 8 NA's in `Renewal.Gross.Step.Up`
#loop through numeric columns and recode NA's to column mean
home_warranty_base[sapply(home_warranty_base, is.numeric)] = lapply(home_warranty_base[sapply(home_warranty_base, is.numeric)], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

#checking again for NA's in dataset
sum(is.na(home_warranty_base)) # 0

#trim number of observations to 5000 and dropping below unimportant columns in order to easier manipulate
#dropping columns: 'Expire.Period','HVAC.Plus.Count','Service.Fee.Saveup.Count','Addt.Frig.Count', 'Wine.Cooler.Count','FS.Freezer.Count','Wet.Bar.Frig.Count',
#                  'Freshwater.Pool'.'Spa.Count','Saltwater.Pool.Spa.Count','Addt.Pool.Spa.Count','Purchased.Septic.Count','Water.Softener.Count',
#                  'PSHVACTU.Count','Pipe.Leak.Count','Heat.Pump.Count','Well.Pump.Count','Purchased.Roof.Count'
set.seed(123)
home_warranty_1 = home_warranty_base %>%
  filter(Expire.Year == 2020) %>% #vast majority of observations fall into year 2020 = 22504/29043 total rows
  sample_n(5000, replace = FALSE) %>% #sampling 5000 observations with set seed value
  dplyr::select(-HVAC.Plus.Count, -Service.Fee.Saveup.Count, -Expire.Period, -Expire.Year) %>% #dropping unimportant purchase options + now year column as it will be factor with 1 class
  dplyr::select(-(Addt.Frig.Count:Purchased.Roof.Count)) %>% #dropping unimportant purchase options
  relocate(Renewed.) #move binary response variable to front of df

#checking new dimension and column names of df
dim(home_warranty_1)
colnames(home_warranty_1)

#converting appropriate columns into factors
home_warranty_1$Primary.Sales.Channel = as.factor(home_warranty_1$Primary.Sales.Channel)
home_warranty_1$Payment.Method = as.factor(home_warranty_1$Payment.Method)
home_warranty_1$Brochure.Name = as.factor(home_warranty_1$Brochure.Name)
home_warranty_1$Expire.Month = as.factor(home_warranty_1$Expire.Month)
home_warranty_1$Sales.Region = as.factor(home_warranty_1$Sales.Region)
home_warranty_1$State = as.factor(home_warranty_1$State)

#recode appropriate factor variables into binary
home_warranty_1$Had.Phone.Number. = ifelse(home_warranty_1$Had.Phone.Number. == "Phone Number", 1, 0)
home_warranty_1$Had.Email.Address. = ifelse(home_warranty_1$Had.Email.Address. == "Email Address", 1, 0)
home_warranty_1$Had.Web.Profile. = ifelse(home_warranty_1$Had.Web.Profile. == "Web Profile", 1, 0)
home_warranty_1$Had.Claim. = ifelse(home_warranty_1$Had.Claim. == "Yes", 1, 0)
home_warranty_1$Purchased.Option. = ifelse(home_warranty_1$Purchased.Option. == "Yes", 1, 0)

#recode step ups to numeric variables
home_warranty_1$Renewal.Discount.Step.Up = suppressWarnings(as.numeric(home_warranty_1$Renewal.Discount.Step.Up))
home_warranty_1$Renewal.Net.Step.Up = suppressWarnings(as.numeric(home_warranty_1$Renewal.Net.Step.Up))

#rename variables for clarity
home_warranty_2 = home_warranty_1 %>%
  rename(Renewed = Renewed., Had.Phone = Had.Phone.Number., Had.Email = Had.Email.Address.,
         Had.Webprofile = Had.Web.Profile., Purchased.Option = Purchased.Option.,
         Number.Of.Options.Purchased = Option.Purchased, Had.Claim = Had.Claim., Condo = Condo.)

#check structure of df
str(home_warranty_2)

#check for class bias in Y variable
table(home_warranty_2$Renewed)



################################################################################################################
#Checking for correlations in data set
################################################################################################################

#look at overall correlation of numeric variables in df
data.correlations = cor(home_warranty_2[sapply(home_warranty_2, is.numeric)]); data.correlations

#### Contact Info ####
#correlation between a client having contact info on file versus the probability of home warranty renewal success
contact.info.sums = home_warranty_2 %>%
  group_by(Renewed) %>%
  summarise_each(funs(sum, n()), Had.Phone, Had.Email, Had.Webprofile); contact.info.sums

#plotting relationship between contact info and home warranty renewal
contact.info.correlation = cor(home_warranty_2[c(1,12,13,14)])

col2 = colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                           "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                           "#4393C3", "#2166AC", "#053061"))
contact.info.correlation.plot = corrplot(contact.info.correlation, col = col2(200), addCoef.col = "dark grey")
mtext("Correlation Matrix of Contact Info vs Home Warranty Renewal", at = 2.5, line = -2, cex = 1.75)

#pairs plot
contact.info.plot.pairs = pairs(contact.info.correlation, col = home_warranty_2$Renewed)

#melting correlation data into df and visualizing with ggplot2
contact.info.melt = reshape2::melt(contact.info.correlation, varnames = paste0("Contact.Info", 1:2), value.name = "correlation")
ggplot(contact.info.melt, aes(Contact.Info1, Contact.Info2, fill = correlation)) + geom_tile() + ggtitle("Correlation across Contact Info On File for Home Warranty Renewal")

#### Housing Market Index and Consumer Sentiment Index by Month ####
#group indexes by month and calculate mean
month.indeces.mean = home_warranty_2 %>%
  group_by(Expire.Month) %>%
  summarise_each(funs(mean, n()), NAHB.Housing.Market.Index, Consumer.Sentiment.Index)

ggplot(month.indeces.mean) + geom_point(aes(x = Expire.Month, y = NAHB.Housing.Market.Index_mean, colour = "NAHB.Housing.Market.Index_mean")) + 
  geom_point(aes(x = Expire.Month, y = Consumer.Sentiment.Index_mean, colour = "Consumer.Sentiment.Index_mean")) + 
  labs(x = "Home Warranty Expiration Month", y = "Indexes", title = "Mean NAHB Housing Market Index + Consumer Sentiment Index by Expiration Month") + theme_dark()
#indexes seem to converge around 75



################################################################################################################
#Logistic Regression Model Full
################################################################################################################

str(home_warranty_2)

#subset uninfluencing factors out of data set
final_df = home_warranty_2 %>%
  dplyr::select(-c(Warranty.key, Brochure.Name, State, Zip.Code, House.over.5.000.Sq.Ft,
            Sales.Region, Sell.Date, Expiration.Date, Base.Price, Option.Price, Had.Claim))

str(final_df)
summary(final_df)

#first logit model using all data points as predictor variables
model_full = glm(Renewed ~ ., data = final_df, family = 'binomial')
summary(model_full)
caret::varImp(model_full)

summary_full_glm = summary(model_full)
Pseudo_R_Full = list(summary_full_glm$coefficients, round(1 - (summary_full_glm$deviance / summary_full_glm$null.deviance), 2)); Pseudo_R_Full

#cross validation of model_full
set.seed(4)
cv_model_full = cv.glm(na.omit(final_df), model_full, K = 10)
cv_model_full$delta

anova(model_full, test = "Chisq")

#Identify overinfluential observations with Cook's distance values
#Scaled change in fitted values, which is useful for identifying outliers in the X predictor values
plot(model_full, which = 4, id.n = 5)

#odds ratio + confidence intervals
exp(cbind(OR = coef(model_full), confint(model_full)))

#Residual deviance: 5847.4 on 4956  degrees of freedom
#AIC: 5921.4
#Delta: 0.2059676 0.2057791

#Evaluation: running the full model, you can clearly the following variables are 
#statistically significant towards predicting home warranty renewal success:
### Supreme.in.Base + Had.Email + Had.Webprofile + Expire.Month + Primary.Sales.Channel + Auto.Renew + Discount + Claim.Count + High.Claims + Purchased.Supreme.Count + 
### Purchased.Wash.Dryer.Count + Number.Of.Options.Purchased + Renewal.Option.Step.Up + Renewal.Discount.Step.Up

#NAs can be explained by linear dependence of the predictor variables. Need to remove these to avoid dummy variable trap
#-----------------------------------------------------------------------------------------------------------



################################################################################################################
#Logistic Regression Using STEPAIC to Fit
#Will drop predictor variables in each model and test AIC, VIF, and Odds Ratio
################################################################################################################

model_1 = stepAIC(model_full, direction="both")
summary(model_1)
caret::varImp(model_1)

sort((vif(model_1)), decreasing = TRUE) #sorting and checking for the highest VIF valued variable

model_2 = glm(Renewed ~ Supreme.in.Base + Had.Email + Had.Webprofile +
                Expire.Month + Primary.Sales.Channel + Auto.Renew +
                Discount + Claim.Count + High.Claims + Purchased.Supreme.Count + 
                Purchased.Wash.Dryer.Count + Number.Of.Options.Purchased + 
                Renewal.Option.Step.Up + Renewal.Discount.Step.Up, family = "binomial", data = final_df)
summary(model_2)

set.seed(4)
cv_model_2 = cv.glm(na.omit(final_df), model_2, K = 10)
cv_model_2$delta

#odds ratio + confidence intervals
exp(cbind(OR = coef(model_2), confint(model_2)))

anova(model_2, test = "Chisq")

#-----------------------------------------------------------------------------------------------------------
#Residual deviance: 5873.0  on 4970  degrees of freedom
#AIC: 5919
#Delta: 0.2057145 0.2056059

#Evaluation: by removing beta coefficients with high p values and statistical insignificance, the AIC and the residual delta have decreased
#-----------------------------------------------------------------------------------------------------------

#model final based off stepped model and removing values with too large of P values from anova test
model_final = glm(Renewed ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                Renewal.Discount.Step.Up, family = "binomial", data = final_df)
summary(model_final)

summary_final_glm = summary(model_final)
#A true measure of fit is one based strictly on a comparison of observed to predicted values from the fitted model
#R2 values in logit are always low
Pseudo_R_Final = list(summary_final_glm$coefficients, round(1 - (summary_final_glm$deviance / summary_final_glm$null.deviance), 2)); Pseudo_R_Final

#cross validation of model_final
set.seed(4)
cv_model_final = cv.glm(na.omit(final_df), model_final, K = 10)
cv_model_final$delta

#odds ratio + confidence intervals
exp(cbind(OR = coef(model_final), confint(model_final)))

#Identify overinfluential observations with Cook's distance values
#Scaled change in fitted values, which is useful for identifying outliers in the X predictor values
plot(model_final, which = 4, id.n = 5)

#showing factors that most influenced our renewal success probability
caret::varImp(model_final)

#comparing models
anova(model_final, model_1, model_full, test = "Chisq")

#small observable class bias in renewed residuals
qqnorm(model_final$residuals)
qqline(model_final$residuals, col=2)
hist(model_final$residuals)

#logistic regression does not assume the residuals are normally distributed nor that the variance is constant. 
#However, the deviance residual is useful for determining if individual points are not well fit by the model. 
#Here we can fit the standardized deviance residuals to see how many exceed 3 standard deviations.
model_final_mutate = augment(model_final) %>%
  mutate(index = 1:n())

ggplot(model_final_mutate, aes(index, .std.resid)) + 
  geom_point(alpha = .5) +
  geom_ref_line(h = 3) + ggtitle("Plot of Standard Deviation Distribution of Residuals")

#No outliers in the residuals (all residual observations within 3 SD)
model_outliers = model_final_mutate %>% 
  filter(abs(.std.resid) > 3)




#-----------------------------------------------------------------------------------------------------------
#Residual deviance: 5856.9  on 4969  degrees of freedom
#AIC: 5904.9
#Delta: 0.2049345 0.2048219

#Final Predictor Variables: `Supreme.in.Base`, `Had.Phone`, `Had.Email`, `Had.Webprofile`, `Expire.Month`, `Primary.Sales.Channel`
#                           `Auto.Renew`, `Discount`, `Claim.Count`, `Service.Fee.Buydown.Count`, `Purchased.Supreme.Count`
#                           `Purchased.Wash.Dryer.Count`, `Renewal.Option.Step.Up`, `Renewal.Discount.Step.Up`

#some interpretation: With all other input variables unchanged, every unit of increase in the client filing a claim 
#increases the odds of the client renewing their home warranty by a factor of 1.609.
################################################################################################################
#Model Evaluation
#Due to the fact that accuracy isn't suitable for this situation, we'll have to use another measurement to decide which cutoff value to choose, the ROC curve.
#The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
################################################################################################################

#break df into training in test set with train = 80% of data and test = 20% of data
set.seed(100)
#split data into training and test sets
indices = sample.split(final_df$Renewed, SplitRatio = 0.8)
train = final_df[indices,]
test = final_df[!(indices),]

#predicted probabilities of success for test data
test_pred = predict(model_final, type = "response", newdata = test[,-1])

# Let's see the summary 
summary(test_pred)

test$prob = test_pred

#using the probability cutoff of 50%
test_pred_success = factor(ifelse(test_pred >= 0.50, "Yes", "No"))
test_actual_success = factor(ifelse(test$Renewed == 1,"Yes","No"))

#at 50% probability cutoff, we achieved 664/1000 correct predictions
table(test_actual_success, test_pred_success)

#checking for other levels of cut off by calibrating the output of the binary classification model and examining all possible outcomes of predictions
#confusion matrix reveals how confused the model is between the different classes
#
# Prediction        Reference
#                 Yes        No
#       Yes   True Pos    False Pos  
#
#       No    False Neg   True Neg

#At 0.40 
test_pred_success_2 = factor(ifelse(test_pred >= 0.40, "Yes", "No")) 
test_conf2 = confusionMatrix(test_pred_success_2, test_actual_success, positive = "Yes"); test_conf2

#At 0.30
test_pred_success_3 = factor(ifelse(test_pred >= 0.30, "Yes", "No")) 
test_conf3 = confusionMatrix(test_pred_success_3, test_actual_success, positive = "Yes"); test_conf3

#At 0.20
test_pred_success_4 = factor(ifelse(test_pred >= 0.20, "Yes", "No")) 
test_conf4 = confusionMatrix(test_pred_success_4, test_actual_success, positive = "Yes"); test_conf4

train$prediction = predict(model_final, newdata = train, type = "response")
test$prediction = predict(model_final, newdata = test, type = "response")
test$real = test$Renewed

#obtaining the predicted value that a client will renew on both training and testing set. 
#After that we'll perform a quick evaluation on the training set by plotting the probability (score) estimated by our model with a double density plot.
#a ideal double density plot you want the distribution of scores to be separated,
#with the score of the negative instances to be on the left and the score of the positive instance to be on the right.
ggplot( train, aes(prediction, color = as.factor(Renewed) ) ) + 
  geom_density( size = 2 ) +
  ggtitle( "Training Set's Predicted Score" ) + 
  scale_color_economist( name = "data", labels = c( "negative", "positive" ) )
#Our skewed double density plot, however, can actually tell us a very important thing: Accuracy will not be a suitable measurement for this model.
#Since the prediction of a logistic regression model is a probability, in order to use it as a classifier, 
#we'll have to choose a cutoff value, or you can say its a threshold value. Where scores above this value will classified as positive, those below as negative.


#ROC curve for this model
#Shows you the trade off between the rate at which you can correctly predict something
#with the rate of incorrectly predicting something when choosing different cutoff values.
#This curve is created by plotting the true positive rate (TPR) on the y axis against the false positive rate (FPR) on the x axis
#at various cutoff settings ( between 0 and 1 ).
#ROC analysis provides tools to select possibly optimal models and to discard suboptimal ones.
#AUC: measure ranging from 0 to 1, shows how well the classification model is performing in general = 74.1%
plot.roc(test$real, test$prediction, col = "red", main = "ROC Validation set",
         percent = TRUE, print.auc = TRUE)

# finding the optimal probability cutoff value
perform_fn = function(cutoff) {
  predicted_success = factor(ifelse(test_pred >= cutoff, "Yes", "No"))
  conf = confusionMatrix(predicted_success, test_actual_success, positive = "Yes")
  acc = conf$overall[1]
  sens = conf$byClass[1]
  spec = conf$byClass[2]
  out = t(as.matrix(c(sens, spec, acc))) 
  colnames(out) = c("sensitivity", "specificity", "accuracy")
  return(out)
}

summary(test_pred)
s = seq(.01,.80,length=1000)
OUT = matrix(0,1000,3)

conf.matrix = data.frame(cbind(s,OUT))

for(i in 1:1000) {
  OUT[i,] = perform_fn(s[i])
} 

#plotting linear convergence of probability cutoff sensitivity, specificity, and accuracy
#allows us to visualize the optimal probability cutoff value
ggplot(conf.matrix) + geom_point(aes(x = s, y = V2, colour = "Sensitivity")) + geom_point(aes(x = s, y = V3, colour = "Specificity")) +
  geom_point(aes(x = s, y = V4, colour = "Accuracy")) + labs(x = "Cutoff", y = "Value", title = "Linear Convergence of Probability Prediction Values in Model")

#exact cuttoff value Where scores above this value will classified as positive, those below as negative.
cutoff = s[which(abs(conf.matrix[,2]-OUT[,3]) < 0.01)] #0.5920220

test_cutoff_success = factor(ifelse(test_pred >= 0.61, "Yes", "No"))

conf_final = confusionMatrix(test_cutoff_success, test_actual_success, positive = "Yes"); conf_final
#Accuracy based upon un-weighted values in binary response class : 0.6802

#Using the IV variables in our final model, the final model can classify between Home Warranty Renewal success and failure at a 74.1% probability rate
#Due to the fact that accuracy isn't suitable for this situation, we used the ROC curve:
#we're making a balance between the false positive rate (FPR) and false negative rate (FNR). The objective function for our model,
#where we're trying to minimize the number of mistakes we're making or so called costs. The ROC curve's purpose is used to visualize and quantify the tradeoff
#we're making between the two measures.

#Using a classification algorithm like logistic regression in this example enabled us to estimate probabilities of Home Warranty renewal in the future.
#As the dataset was unbalanced, meaning there were a lot more positive outcome data than negative, we shouldn't use accuracy as the measurement to evaluate our model's performance.

################################################################################################################
#Advanced Machine Learning Algorithms
#We felt most comfortable with our model projections above, but also performed advanced machine learning algorithms to 
#test if more accurate models exist
################################################################################################################

# advanced machine learning algorithms
# Load CART packages
library(rpart)
# install rpart package
#install.packages("rpart.plot")
library(rpart.plot)

#Omit NA values
final_df2 = na.omit(final_df)
final_df2
#break df into training in test set with train = 80% of data and test = 20% of data
set.seed(100)
#split data into training and test sets
indices2 = sample.split(final_df2$Renewed, SplitRatio = 0.8)
train2 = final_df2[indices2,]
test2 = final_df2[!(indices2),]

train2
test2

#Change Renewed into a factor
train2$RenewedFactor = as.factor(train2$Renewed)
test2$RenewedFactor = as.factor(test2$Renewed)
final_df2$RenewedFactor = as.factor(final_df2$Renewed)


#Check that the train and test data are similar
prop.table(table(train2$RenewedFactor))
prop.table(table(test2$RenewedFactor)) 
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# decision tree
set.seed(7)
fit.dt = train(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                 Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                 Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                 Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                 Renewal.Discount.Step.Up, data=train2, na.action=na.omit, method="rpart",metric=metric, trControl=control)
fit.dt
#cp          Accuracy   Kappa    
#0.02973978  0.6990520  0.3746903

# CART model
renewedtree = rpart(Renewed ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                      Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                      Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                      Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                      Renewal.Discount.Step.Up,   data=final_df2)

# Plot the tree using prp command defined in rpart.plot package
prp(renewedtree)

# Random Forest
set.seed(7)
fit.rf = train(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                  Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                  Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                  Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                  Renewal.Discount.Step.Up, data=train2, na.action=na.omit, method="rf",metric=metric, trControl=control)
fit.rf
#mtry  Accuracy   Kappa    
#12    0.7821792  0.5494473

#library(randomForest)
fit = randomForest(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                      Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                      Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                      Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                      Renewal.Discount.Step.Up, na.action=na.omit,  data=final_df2)

fit 
importance(fit) # importance of each predictor
#Rank Variables in term of Importance: Renewal Discount Step Up, Renewal Option Step Up, Discount, Expire Month, Claim Count,
#Auto Renew, Had Email, Purchase Supreme Count, Primary Sales Channel, High Claims, Had Web Profile, Service Fee Buydown Count, 
#Supreme in Base, Purchased Washer/Dryer, Had Phone

#                            MeanDecreaseGini
#Supreme.in.Base                    19.81542
#Had.Email                          85.07460
#Had.Phone                          16.30914
#Had.Webprofile                     29.40951
#Expire.Month                      180.76829
#Primary.Sales.Channel              34.91059
#Auto.Renew                         87.12413
#Discount                          215.05757
#Claim.Count                       113.14630
#High.Claims                        30.60089
#Service.Fee.Buydown.Count          27.75447
#Purchased.Supreme.Count            65.60800
#Purchased.Wash.Dryer.Count         15.16460
#Renewal.Option.Step.Up            355.45511
#Renewal.Discount.Step.Up          460.60145


# kNN
set.seed(7)
fit.knn = train(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                   Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                   Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                   Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                   Renewal.Discount.Step.Up, data=train2, na.action=na.omit, method="knn",metric=metric, trControl=control)
fit.knn
# k  Accuracy   Kappa    
#9  0.7253503  0.4392279

# SVM
set.seed(7)
fit.svm = train(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                   Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                   Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                   Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                   Renewal.Discount.Step.Up, data=train2, na.action=na.omit, method="svmRadial",metric=metric, trControl=control)
fit.svm

#neural net
set.seed(7)
fit.nn = train(RenewedFactor ~ Supreme.in.Base + Had.Email + Had.Phone + Had.Webprofile +
                  Expire.Month + Primary.Sales.Channel + Auto.Renew + 
                  Discount + Claim.Count + High.Claims + Service.Fee.Buydown.Count + 
                  Purchased.Supreme.Count + Purchased.Wash.Dryer.Count + Renewal.Option.Step.Up + 
                  Renewal.Discount.Step.Up, data=train2, na.action=na.omit, method="nnet",metric=metric, trControl=control)
fit.nn
# size  decay  Accuracy   Kappa
# 3     1e-01  0.6925332  0.38568029

# summarize accuracy of models
results = resamples(list(cart=fit.dt, knn=fit.knn, svm=fit.svm, rf=fit.rf,nn =fit.nn))
summary(results)
#making prediction
predictionsdt = predict(fit.dt, test2)
confusionMatrix(predictionsdt, test2$RenewedFactor)
#balanced accuracy 67.15%

predictionsrf = predict(fit.rf, test2)
confusionMatrix(predictionsrf, test2$RenewedFactor)
#balanced accuracy 75.53%

predictionsknn = predict(fit.knn, test2)
confusionMatrix(predictionsknn, test2$RenewedFactor)
#balanced accuracy 68.88%

predictionssvm = predict(fit.svm, test2)
confusionMatrix(predictionssvm, test2$RenewedFactor)
#balanced accuracy 66.22%

predictionsnn = predict(fit.nn, test2)
confusionMatrix(predictionsnn, test2$RenewedFactor)
#Balanced accuracy 74.23%

#Neural net and random forest built the most accurate models in machine learning and to predict renewal rate.
#Random forest had the highest accuracy at a balanced accuracy of 75.53%.  As this improved the model accuracy by a percentage point than with using the ROC
#curve to predict renewal rates we used RF to describe our final model.
