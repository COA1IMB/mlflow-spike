import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
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

      # transform features: missing value handling and one hot encoder
      enum_features = credit.select_dtypes(include=['object']).columns
      numeric_features = credit.select_dtypes(exclude=['object']).columns

      numeric_transformer = Pipeline(
          steps=[('missing_imputer',
                  SimpleImputer(strategy='constant', fill_value=float('-inf')))])
      enum_tramsformer = Pipeline(
          steps=[('missing_imputer',
                  SimpleImputer(strategy='constant', fill_value="NaN")),
                 ('encoding',
                  preprocessing.OneHotEncoder(handle_unknown='ignore'))])

      preprocessor = ColumnTransformer(
          transformers=[
              ('numeric_transformer', numeric_transformer, numeric_features),
              ('enum_tramsformer', enum_tramsformer, enum_features)])

      # define and train final model
      rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                  min_samples_leaf=3, random_state=42)
      pipe_model_rf = Pipeline([('prepocessor', preprocessor), ('ml_model', rf)])
      pipe_model_rf = pipe_model_rf.fit(credit, label)

      mlflow.sklearn.log_model(pipe_model_rf, "model", registered_model_name="Credit")
