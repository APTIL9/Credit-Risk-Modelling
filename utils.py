import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

def clean_data(df):
    # Fill missing values with median or mode
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].mode()[0], inplace=True)

    # Drop unnecessary columns
    df.drop(['cb_person_default_on_file'], axis=1, inplace=True)

    return df

def explore_data(df):
    # Plot histograms of numerical variables
    df.hist(bins=20, figsize=(15, 10))
    plt.show()

    # Plot scatterplots of numerical variables
    sns.pairplot(df, hue='loan_status')
    plt.show()

    # Plot correlation matrix
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def train_models(df):
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop('loan_status', axis=1),
                                                        df_encoded['loan_status'],
                                                        test_size=0.2,
                                                        random_state=42)

    # Train logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Train random forests model
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb_model.fit(X_train, y_train)

    # Make predictions on test set
    lr_pred = lr_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # Evaluate models using ROC-AUC score and plot ROC curves
    evaluate_models(X_test, y_test, lr_pred, rf_pred, xgb_pred)
 
def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC-ROC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def evaluate_models(X_test, y_test, lr_pred, rf_pred, xgb_pred):
    # Calculate ROC-AUC score for each model
    lr_auc = roc_auc_score(y_test, lr_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred)

    # Print ROC-AUC scores
    print(f"Logistic Regression ROC-AUC score: {lr_auc:.3f}")
    print(f"Random Forests ROC-AUC score: {rf_auc:.3f}")
    print(f"XGBoost ROC-AUC score: {xgb_auc:.3f}")

    # Plot ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred)

    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forests (AUC = {rf_auc:.3f})")
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
