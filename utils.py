import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def clean_data(df):
    """
    Clean and preprocess the loan data.

    Parameters:
    df (pandas.DataFrame): the loan dataframe

    Returns:
    pandas.DataFrame: the cleaned loan dataframe
    """
    # Drop Loan ID and Customer ID columns
    df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)

    # Convert Current Loan Amount and Credit Score to numeric data type
    df['Current Loan Amount'] = pd.to_numeric(df['Current Loan Amount'], errors='coerce')
    df['Credit Score'] = pd.to_numeric(df['Credit Score'], errors='coerce')

    # Fill missing values in Current Loan Amount and Credit Score with mean
    df['Current Loan Amount'].fillna(df['Current Loan Amount'].mean(), inplace=True)
    df['Credit Score'].fillna(df['Credit Score'].mean(), inplace=True)

    # Replace missing values in Months since last delinquent with 0
    df['Months since last delinquent'].fillna(0, inplace=True)

    # Convert Years in current job to numeric data type and replace missing values with mode
    df['Years in current job'] = pd.to_numeric(df['Years in current job'].str.extract('(\d+)'), errors='coerce')
    df['Years in current job'].fillna(df['Years in current job'].mode()[0], inplace=True)

    # Replace missing values in Bankruptcies and Tax Liens with 0
    df['Bankruptcies'].fillna(0, inplace=True)
    df['Tax Liens'].fillna(0, inplace=True)

    return df

def eda(df):
    """
    Perform exploratory data analysis on the loan data.

    Parameters:
    df (pandas.DataFrame): the loan dataframe

    Returns:
    pandas.DataFrame: the exploratory data analysis dataframe
    """
    # Create a new dataframe for EDA
    eda_df = pd.DataFrame()

    # Add Loan Status value counts to EDA dataframe
    eda_df['Loan Status'] = df['Loan Status'].value_counts()

    # Plot a histogram of Current Loan Amount
    plt.hist(df['Current Loan Amount'], bins=20)
    plt.xlabel('Current Loan Amount')
    plt.ylabel('Count')
    plt.title('Histogram of Current Loan Amount')
    plt.show()

    # Plot a boxplot of Credit Score
    sns.boxplot(df['Credit Score'])
    plt.xlabel('Credit Score')
    plt.title('Boxplot of Credit Score')
    plt.show()

    # Add mean and median of Credit Score to EDA dataframe
    eda_df.loc['Credit Score Mean'] = df['Credit Score'].mean()
    eda_df.loc['Credit Score Median'] = df['Credit Score'].median()

    # Add Home Ownership value counts to EDA dataframe
    eda_df['Home Ownership'] = df['Home Ownership'].value_counts()

    return eda_df

def plot_roc_curve(model, X_test, y_test, model_name):
    # predict probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # plot ROC curve
    plt.plot(fpr, tpr, lw=2, alpha=0.8,
             label='%s (AUC = %0.2f)' % (model_name, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random Chance', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

