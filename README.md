# Credit Risk Modeling Project
This project aims to develop a credit risk model that predicts whether customers will default on their loans using machine learning techniques. The dataset used for this project consists of various attributes related to customers and their loan information.

## Getting Started
To get started with this project, clone the repository to your local machine and ensure that the necessary dependencies are installed. The following packages are required for this project:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

## Data Cleaning and Preprocessing
The data is first cleaned and preprocessed using various techniques such as removing duplicates, handling missing values, and transforming variables. This ensures that the data is ready for analysis and modeling.

## Exploratory Data Analysis
Exploratory data analysis (EDA) is performed to gain insights into the data and identify key variables that impact credit risk. EDA is conducted using data visualization techniques to identify patterns and relationships in the data.

![image](https://user-images.githubusercontent.com/126561743/221992462-060635e7-4480-432a-9eab-565f4265c57e.png)

## Model Development
Three machine learning models are developed and implemented to accurately predict credit risk for customers: logistic regression, random forests, and XGBoost. These models are trained on the preprocessed data and evaluated using various performance metrics.

## Results
The random forests model achieved the highest score of 0.936, XGBoost model with a AUC-ROC score of 0.93, followed by the logistic regression model with a score of 0.89, and the logistic regression model with a score of 0.752.

![image](https://user-images.githubusercontent.com/126561743/221989982-4a3e66c8-39aa-4755-82a1-e0254a0f5388.png)

## Conclusion
This project demonstrates the use of machine learning techniques to develop a credit risk model that can accurately predict whether customers will default on their loans. The results show that random forests is the most effective model for this task, with a high AUC-ROC score. This project can be used as a reference for anyone looking to build a similar model in the finance industry.



