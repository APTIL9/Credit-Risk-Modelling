from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def credit_risk_model(df):
    """
    Build and evaluate a credit risk model using logistic regression, random forests, and XGBoost.

    Parameters:
    df (pandas.DataFrame): the loan dataframe

    Returns:
    None
    """

    # Preprocessing the data
    df = data_preprocessing(df)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Building logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Building random forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Building XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # Evaluating the models using AUC-ROC score
    lr_score = roc_auc_score(y_test, lr_preds)
    rf_score = roc_auc_score(y_test, rf_preds)
    xgb_score = roc_auc_score(y_test, xgb_preds)

    # Printing the AUC-ROC scores
    print("Logistic Regression AUC-ROC score:", lr_score)
    print("Random Forest AUC-ROC score:", rf_score)
    print("XGBoost AUC-ROC score:", xgb_score)
    
    # Plotting the ROC curves
    plot_roc_curve(lr_model, X_test, y_test, "Logistic Regression")
    plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
    plot_roc_curve(xgb_model, X_test, y_test, "XGBoost")
    
   

if __name__ == "__main__":
  df_data = pd.read_csv('dataset_credit.csv')
  credit_risk_model(df_data)
