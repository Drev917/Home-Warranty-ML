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
