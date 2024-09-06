import os
import numpy as np
from src.logger import logging
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.utils import evaluate_models
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # The last column is 'Sleep Disorder'
            X_train = train_array[:, :-1]  # All columns except the last one
            y_train = train_array[:, -1]   # Last column (target variable)
            X_test = test_array[:, :-1]    # All columns except the last one
            y_test = test_array[:, -1]     # Last column (target variable)

            logging.info("Data splitting completed successfully")
            
            classification_models = {
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "Logistic Regression": LogisticRegression()
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=classification_models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = classification_models[best_model_name]
            
            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            f1 = f1_score(y_test,predicted,average='weighted')
            return r2_square,f1
        
        
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise
