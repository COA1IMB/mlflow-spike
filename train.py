import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import mlflow
from urllib.parse import urlparse
import mlflow.sklearn

''' Required versions to execute this example
pandas: 1.2.4
scitkit-learn: 0.24.2
'''
if __name__ == "__main__":
      credit = pd.read_csv("credit.csv", delimiter=';')

      # drop id and societal questionable features
      credit = credit.drop(['id', 'gender'], axis=1)

      # encode response column
      le_response_column = preprocessing.LabelEncoder()
      le_response_column = le_response_column.fit(credit.defaulted)
      label = le_response_column.transform(credit.defaulted)
      credit.drop(['defaulted'], axis=1, inplace=True)

      with mlflow.start_run():

        # define and train final model
        rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                    min_samples_leaf=3, random_state=42)
        rf.fit(credit, label)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
          mlflow.sklearn.log_model(rf, "model", registered_model_name="Credit")
        else:
          mlflow.sklearn.log_model(rf, "model")