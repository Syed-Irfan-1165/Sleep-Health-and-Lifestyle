import os
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                                         'Stress Level', 'Heart Rate', 'Daily Steps', 'High_BP', 'Low_BP'])
        
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[col] = np.clip(X[col], lower_bound, upper_bound)
        return X.values

class LabelEncoderTransformer:
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        self.encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col, encoder in self.encoders.items():
            # Handle unseen categories
            X_encoded[col] = X_encoded[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        return X_encoded


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                              'Stress Level', 'Heart Rate', 'Daily Steps', 'High_BP', 'Low_BP']
            categorical_cols = ['Gender', 'Occupation', 'BMI Category']

            def replace_normal_weight(X):
                X = X.copy()
                if 'BMI Category' in X.columns:
                    X['BMI Category'] = X['BMI Category'].replace("Normal Weight", "Normal")
                return X

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("outlier_handler", OutlierHandler()),
                    
                ]
            )
            logging.info("Numerical Columns Standardization completed.")
            logging.info("Outliers been taken care of.")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("replace_normal_weight", FunctionTransformer(replace_normal_weight, validate=False)),
                    ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
            )
            logging.info("Categorical Column BMI_Category 'Normal Weights' converted to 'Normal' successfully")
            logging.info("BMI Category successfully encoded using Ordinal Encoder.")

            gender_occupation_pipeline = Pipeline([
                ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            logging.info("Categorical Columns 'Gender' & 'Occupation' Encoded successfully.")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("bmi_pipeline", cat_pipeline, ['BMI Category']),
                ("gender_occupation_pipeline", gender_occupation_pipeline, ['Gender', 'Occupation'])
            ])
            
            logging.info(f"Standard scaling of Categorical columns completed: {categorical_cols}")
            logging.info(f"Encoding of Numerical columns completed: {numerical_cols}")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            print("Train DataFrame Columns:", train_df.columns)
            print("Test DataFrame Columns:", test_df.columns)

            for df in [train_df, test_df]:
                if 'Blood Pressure' in df.columns:
                    df[['High_BP', 'Low_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
                    df.drop('Blood Pressure', axis=1, inplace=True)
                else:
                    raise ValueError("Column 'Blood Pressure' not found in the DataFrame.")

            X_train = train_df.drop(columns=['Person ID', 'Sleep Disorder'])
            y_train = train_df['Sleep Disorder']
            
            X_test = test_df.drop(columns=['Person ID', 'Sleep Disorder'])
            y_test = test_df['Sleep Disorder']

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            preprocessing_obj = self.get_data_transformer_object()
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            print("X_train Columns before transformation:", X_train.columns)
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            print("X_train after transformation:", X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            print("X_train_transformed shape:", X_train_transformed.shape)
            print("X_test_transformed shape:", X_test_transformed.shape)
            
            logging.info(f"Saving preprocessing object.")
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessing_obj
            
        except Exception as e:
            raise CustomException(e, sys)
