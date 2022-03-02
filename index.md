## Welcome to DSC180B Section B06's Explainable AI Webpage!

### Authors

- [Jerry (Yung-Chieh) Chan](https://github.com/JerryYC)
- [Apoorv Pochiraju](https://github.com/apochira)
- [Zhendong Wang](https://github.com/zhw005)
- [Yujie Zhang](https://github.com/yujiezhang0914)

## Introduction - Assess the Fairness and reasoning of Models by Explaining Black Box Model Outcomes using Explainable AI 

Nowadays, the algorithmic decision-making system has been very common in people’s daily lives. Gradually, some algorithms become too complex for humans to interpret, such as some black-box machine learning models and deep neural networks. 

In order to assess the fairness of the models and make them better tools for different parties, we need explainable AI (XAI) to uncover the reasoning behind the predictions made by those black-box models. 

In the project, we will be focusing on using different techniques from causal inferences and explainable AI to interpret various classification models across various domains. In particular, we are interested in three domains - healthcare, finance, and the housing market. Within each domain, we are going to train four binary classification models first, and we have four goals in general: 
1) Explaining black-box models both globally and locally with various XAI methods; 
2) Assessing the fairness of each learning algorithm with regard to different sensitive attributes; 
3) Generating recourse for individuals - a set of minimal actions to change the prediction of those black-box models;
4) Evaluating the explanations from those XAI methods using domain knowledge.


## Datasets 

In our project, we use datasets from three domains: Healthcare, Finance, and Housing Market.

### Health Care

We choose healthcare as one of the domains because it is sensitive and assessing fairness of decision algorithms on this domain is important. For the healthcare domain, we will use the data on hospital readmission for diabetes patients obtained from Kaggle. The data was collected for the Hospital Readmission Reduction Program operated by Medicare & Medicaid Services and indicates whether diabetic patients were readmitted to the hospital and whether it was within 30 days or after beyond 30 days. This dataset contains records for 101,766 patients and includes attributes for each patient such as race, age, gender, insulin levels, type of admission, and other specific information about medical history. For this dataset, we will predict whether the patient will be readmitted.

[LINK to Health Care Dataset](https://www.kaggle.com/iabhishekofficial/prediction-on-hospital-readmission)

### Finance

We choose finance domain because our life is filled with topics related to finance and money, so it is important and interesting to find out the reasoning behind machine learning model decisions in this domain. For the finance domain, we will use the loan defaulter dataset obtained from Kaggle. The loan defaulter dataset consists of information such as gender and income of 307,511 applicants. For this dataset, we will predict whether an applicant will be a defaulter.

[LINK to Finance Dataset](https://www.kaggle.com/gauravduttakiit/loan-defaulter)

### Housing

We choose the housing market domain because housing is an essential element in everyone's life as well. For the housing market domain, we will use the Airbnb dataset obtained from Kaggle. The Airbnb dataset consists of basic information such as name and location for 3,818 Airbnb properties. For this dataset, we will predict the class that the price of an Airbnb property falls into.

[LINK to Housing Dataset](https://www.kaggle.com/airbnb/seattle?select=listings.csv)

## Method

We selected four popular machine learning models that's often used in tabular data classification. Our model selection covers classic machine learning model, ensemble model, and deep learning model. Following is a breif introduction to those models. Since this project focuses on model explanation, we will skip the model training details here.

* SVM: Support vector machine is a supervise learning method effective in high dimensional spaces. It project datapoints to high dimensional space and seperate them with a hyperplane.
* LGBM: LightGBM is an ensemble model with gradient boosted decision trees.
* XGBoost: Similar as LGBM, extreme Gradient Boosting is an implementation of gradient boosted decision trees with more regularized parallel decision trees.
* TabNet: TabNet is an deep tabular data learning architecture based on attentive transformer.

### Global Explanation
Global methods describe the average behavior of a machine learning model. 
Introduce global explanation methods PDP...

### Local Explanation
Local interpretation methods explain individual predictions. 
Introduce local explanation method SHAP and LIME

### Counterfactual Explanations
Intorduce counterfactual method...

### Fairness Analysis
We conduct some common fairness tests on the models trained on the loan dataset and the healthcare dataset. We pick out sensitive attributes that should be independent of the target variable base on human knowledge. For the loan dataset, the sensitive variable that we chose is gender, and for the healthcare dataset, the sensitive attribute is race.
For each classification model, we evaluated the fairness based on four definitions: Group Fairness, Predictive Parity, False Positive Error Rate Balance, and False Negative Error Rate Balance. They evaluate the fairness of a model based on different definitions. 
<details>
<summary>Learn more about the four fairness definitions</summary>
<br>

- **Group Fairness**: Check if subjects in both protected and unprotected groups have equal probability of being assigned to the positive predicted class 

- **Predictive Parity**: Check if both protected and unprotected groups have equal PPV – the probability of a subject with positive predictive value to truly belong to the positive class 
  
- **False Positive Error Rate Balance (Predictive Equality)**: Check if both protected and unprotected groups have equal FPR – the probability of a subject in the negative class to have a positive predictive value. 
  
- **False Negative Error Rate Balance (Equal Opportunity)**: Check if both protected and unprotected groups have equal FNR – the probability of a subject in a positive class to have a negative predictive value. 
</details>
 

## Results
- We plan to describe our results in this section using words and plots.

### Global Explanation
some example of global explanation result (plots)

### Local Explanation
Pick three data instance from each dataset...
some example of global explanation of local explanation result (plots) Interactive js code from SHAP and LIME

### Counterfactual
Use the same data instance to generate counterfactual result

### Fairness Analysis
Discuss Fairness Analysis result

### Explanation Comparison
Compare explanation generated from different model and different XAI methods.


## Discussion
- Discuss the results.

## Code
- [LINK](https://github.com/zhw005/DSC180B-Project) to our project github repo. 


## Reference

- Molnar, C. (2021, November 11). Interpretable machine learning. 9.5 Shapley Values. Retrieved December 2, 2021, from https://christophm.github.io/interpretable-ml-book/shapley.html.

