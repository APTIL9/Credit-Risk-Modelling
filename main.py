from utils import *    

if __name__ == "__main__":
  df_data = pd.read_csv('credit_risk_dataset.csv')
  df_cleaned = clean_data(df_data)
  explore_data(df_cleaned)
  train_models(df_cleaned)
