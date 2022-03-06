## Welcome to DSC180B Section B06's Explainable AI Webpage!

### Authors

- [Jerry (Yung-Chieh) Chan](https://github.com/JerryYC)
- [Apoorv Pochiraju](https://github.com/apochira)
- [Zhendong Wang](https://github.com/zhw005)
- [Yujie Zhang](https://github.com/yujiezhang0914)

## Introduction - Assess the Fairness and reasoning of Black Box Model Outcomes using Explainable AI 

Nowadays, the algorithmic decision-making system has been very common in people’s daily lives. Gradually, some algorithms become too complex for humans to interpret, such as some black-box machine learning models and deep neural networks. 

In order to assess the fairness of the models and make them better tools for different parties, we need explainable AI (XAI) to uncover the reasoning behind the predictions made by those black-box models. 

In the project, we are focusing on using different techniques from causal inferences and XAI to interpret various classification models across various domains. In particular, we are interested in three domains - healthcare, finance, and the housing market. Within each domain, we are going to train four binary classification models first, and we have four goals in general: 

1) Explaining black-box models both globally and locally with various XAI methods;

2) Assessing the fairness of each learning algorithm with regard to different sensitive attributes;
 
3) Generating recourse for individuals - a set of minimal actions to change the prediction of those black-box models;
 
4) Evaluating the explanations from those XAI methods using domain knowledge.


## Datasets in Three Domains

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

## Methods of XAI

We selected four popular machine learning models that's often used in tabular data classification. Our model selection covers classic machine learning model, ensemble model, and deep learning model. Following is a breif introduction to those models. Since this project focuses on model explanation, we will skip the model training details here.

* SVM: Support vector machine is a supervise learning method effective in high dimensional spaces. It project datapoints to high dimensional space and seperate them with a hyperplane.
* LGBM: LightGBM is an ensemble model with gradient boosted decision trees.
* XGBoost: Similar as LGBM, extreme Gradient Boosting is an implementation of gradient boosted decision trees with more regularized parallel decision trees.
* TabNet: TabNet is an deep tabular data learning architecture based on attentive transformer.

### Global Explanation
Global methods describe the average behavior of a machine learning model. In this project, we use two global explanation methods: partial dependence plot and Permutation Feature Importance.

The **Partial Dependence Plot (PDP)** works by marginalizing the machine learning model output over the distribution of the features in set C so that the function shows the relationship between the features in set S we are interested in and the predicted outcome. In the case that features are uncorrelated, a PDP shows how the average prediction in the dataset changes when a feature changes. 

The **Permutation Feature Importance** calculates the feature importance by permuting the feature in the dataset. A feature is important if the error of a model increases significantly after permuting the feature. A feature is not important if the error of a model does not change after shuffling the feature values. The Permutation Feature Importance method takes into account all interactions with other features by destroying the interaction effects with other features when permuting the feature. 

<details>
<summary>--> Click to learn more about PDP and Permutation Feature Importance</summary>
<br>

**Partial Dependence Plot**
  
The PDP works by marginalizing the machine learning model output over the distribution of the features in set C, so that the function shows the relationship between the features in set S we are interested in and the predicted outcome. Since the more important the feature is the more varied a PDP is, the numerical feature importance can be defined as the deviation of each unique feature value from the average curve. For categorical features, the importance is defined as the range of the PDP values for the unique categories divided by four, which is the range rule. The PDP can be estimated by calculating averages in the training data. An assumption of the PDP is that the features in set C are not correlated with the features in set S. And the PDP only has a causal interpretation when the features are independent of each other. The PDP is easy to implement and compute. It also has a clear interpretation: in the case that features are uncorrelated, it shows how the average prediction in the dataset changes when a feature changes. 
  
**Permutation Feature Importance**
  
A feature is important if the error of a model increases significantly after permuting the feature. A feature is not important if the error of a model does not change after shuffling the feature values. The general algorithm to calculate permutation feature importance is as follows: 
1. Calculate the original model error.
2. For each feature i, 
- Generate a new feature matrix X by shuffling the values in feature i.
- Calculate the error after the permutation.
- Calculate the difference FI between the original error and the error after the permutation.
3. Sort features by the difference in descending order.
The advantage of this method is that it takes into account all interactions with other features because it destroys the interaction effects with other features when permuting the feature. Also, it is a straightforward method since it does not require retraining the model. 
</details>

### Local Explanation
Local interpretation methods explain individual predictions. 
Introduce local explanation method SHAP and LIME

### Counterfactual Explanations
A counterfactual explanation describes a causal situation in the form of “If X had not occurred, Y would not have occurred.” It's aim to provide percise actions to achieve a desired outcome. For example, if someone was denied for a loan application by some black-box machine learning algorithm, counterfactual explanation can provide them actions they can do to increase their chance of getting the loan. A good counterfactual explanation method should provide multiple, diverse, and realistic counterfactual explanations that produce the predefined prediction as closely as possible. In our project, we are using the state-of-the-art XAI method LEWIS to generate recourses.


[Click here to learn more about LEWIS](https://arxiv.org/pdf/2103.11972.pdf)


### Fairness Analysis
We conduct some common fairness tests on the models trained on the loan dataset and the healthcare dataset. We pick out sensitive attributes that should be independent of the target variable base on human knowledge. For the loan dataset, the sensitive variable that we chose is gender, and for the healthcare dataset, the sensitive attribute is race.
For each classification model, we evaluated the fairness based on four definitions: Group Fairness, Predictive Parity, Matching conditional frequencies, and Causal discrimination. They evaluate the fairness of a model based on different definitions. 


<details>
<summary>--> Click to learn more about the four fairness definitions</summary>
<br>

- **Group Fairness**: A fair model’s prediction should be independent of the sensitive attribute. Therefore, it should have the same probability of giving a positive prediction for individuals of different protected classes. In this test we check if  P(Y = 1|S = si) = P(Y = 1|S = sj). Notice that this test does not require the actual value of the target variable. In other words, the test is independent of whether the model makes the correct predictions.
 
- **Predictive Parity**: This test measures the model’s predictive power on different groups of the protected class. The probability of the model making the correct prediction should be the same across different groups. In this test, we check if the true positive rates are the same among groups: P(T = 1|Y = 1, S= si) = P(T = 1|Y = 1, S = sj). The method can be also be applied with different prediction evaluation metrics.
  
- **Matching conditional frequencies**: This test is similar to the predictive parity test, except we consider the distribution of predicted probabilities rather than the binarized prediction. We binned the predicted probabilities and compare the frequencies of each bin across different groups. For each bin, We check if P(T = 1|Y  ∈  bink, S= si) = P(T = 1|Y  ∈ bink, S = sj)

  
- **Causal discrimination**: The model is counterfactually fair if its prediction is independent of the change of sensitive variable. We conduct the test by flipping or randomly shuffling the sensitive attribute of the test set and checking if the prediction remains the same.
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
[Click here to see code for this project](https://github.com/zhw005/DSC180B-Project)


## Reference

- Molnar, C. (2021, November 11). Interpretable machine learning. 9.5 Shapley Values. Retrieved December 2, 2021, from https://christophm.github.io/interpretable-ml-book/shapley.html.

