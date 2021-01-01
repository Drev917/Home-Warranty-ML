## Home-Warranty-ML
##### Using advanced machine learning algorithms to predict Home Warranty Renewal success.

### EXECUTIVE SUMMARY

#### Major Findings

The sample analyzed consisted of 29,043 home warranty contracts that expired between October 2019 and September 2020.   After a home warranty contract expired, a customer renewed on average 59.8%.  November had the highest renewal rate of 61.8% while December had the lowest renewal rate of 57.3%.  (see Exhibit 1).

Exhibit 1: Renewal Rate by Month Expired

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Renewal%20Rate.JPG)

According to the data, the more engagement from the customer through purchasing optional coverage, selecting autorenewal and filing a claim, the higher renewal rate was observed. 
However, there was poor correlation between renewal rate and any factor provided in the sample.  This includes: NAHB Housing Market Index and the Consumer Market Sentiment Index.
Upon renewal, the average gross price increased $166 with an average discount of $28 for a total net price increase of $138.  

A model of determining renewal rate probability was established using the following factors: supreme in base, email provided, phone provided, web profile established, expired month, sales channel, autorenew selected, discount, claim count, marked high claims, service fee buydown purchased, supreme purchased, washer/dry purchased, option price step up and discount step up. 

##### The model produces the ability to classify a Home Warranty Renewal Success at approximately 76% probability. 

#### Recommendations for Action
The most influential factors in predicting renewal rate are due to customer behaviors and coverage, therefore, we recommend that XYZ Company:
- Preselect autorenew at checkout, with a discount, to force the customer to actively deselect autorenew to decline.  
- Require contact information when original warranty is set up to increase the number of channels to market and sell the renewal to the customer. 
- Influence the customer to purchase service fee buydown and washer/dryer by changing the bonus structure for sales team to incentivize discounts tied to one of the listed options instead of a straight cash discount. 
- Include supreme coverage within the standard coverage. 

#### Analytical Overview 
The provided historical renewal rates by home warranty contracts were examined individually, by month and against each factor to determine if any patterns or associations could be identified. The provided sample was then cleaned to exclude NA values, recode, convert to factors and rename for enhanced clarity.  Multiple statistical models were then created and analyzed to determine the model most likely to accurately predict whether a customer renews or not. 

### PROJECT PRESENTATION
#### Models used:
- KNN
- SVM
- Decision Tree
- Random Forest
- Neural Network
- Logistic Regression

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%201.JPG)

In order to provide a logical step-through of the data we provide an agenda for the analysis:

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%202.JPG)

We have already examined the Executive Summary, so let's get into the data. Below variables are immediately identified as being influential to the overall renewal success.

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%205.JPG)

We used an assortment of libraries to cleanse, preprocess, shape, and develope ML models on the data.

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%206.JPG)

The original data set had over 100k observations. In order to get a sample statistic from which we can make predictions about the population:
- Identified all NA values
  - Recoded these observations with the column mean
- Trimmed observations to a random dataset of 5k rows with a seed number for replication

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%207.JPG)

It's important to check for class bias in your response variable when you are determining accuracy as a metric for model success
![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%208.JPG)

Running cross validations, anova tests, odds ratio and confidence interval comparison, and graphing Cook's distance values to determine optimal predictor variables

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%209.JPG)

Used Stewise AIC in both directions to eliminate multicollinearity and simplify the model. This also reduces the standard errors of the residuals

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2011.JPG)

The final model consists of the following Independent Variables:
- `Supreme.in.Base`
- `Had.Email`
- `Had.Phone`
- `Had.Webprofile`
- `Expire.Month`
- `Primary.Sales.Channel`
- `Auto.Renew`
- `Discount`
- `Claim.Count`
- `High.Claims`
- `Service.Fee.Buydown.Count`
- `Purchased.Supreme.Count`
- `Purchased.Wash.Dryer.Count`
- `Renewal.Option.Step.Up`
- `Renewal.Discount.Step.Up`

Also worth noting that by plotting the standard deviation distribution of the residuals, all align < 3 SD

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2012.JPG)

Looked at Pseudo Rsquared, Confusion Matrices, and ROC AUC to annalyze logistic regression model

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2014.JPG)

Wrote a function to locate the optimal probability cutoff value at the intersection of accuracy, sensitivity, and specificity

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2016.JPG)

The Receiver Operator Characteristic curve identifies the tradeoff between true positives and false positives

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2017.JPG)

Now on to the Advanced Machine Learning Models!

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2018.JPG)

Decision Tree: to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves)

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2020.JPG)

Random Forest to build off the decision tree algo

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2021.JPG)

K-Nearest Neighbors and Neural Net

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2022.JPG)

Random Forest had the highest balanced accuracy of prediction at 75.53%

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2023.JPG)

#### Recommendations for Action to management

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2024.JPG)

Thanks for reading!

![ScreenShot](https://github.com/Drev917/Home-Warranty-ML/blob/main/Slides/Slide%2025.JPG)
