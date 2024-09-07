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
            
            logging.info("Evaluating classification models")
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=classification_models
            )
            
            best_model_score = max(model_report.values())
            
            best_model_name = max(model_report,key=model_report.get)
            
            best_model = classification_models[best_model_name]
            
            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best found model: {best_model_name} with F1 Score: {best_model_score}")
            
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            # r2 = r2_score(y_test, y_pred)
            logging.info(f"Model Evaluation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1},")
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
        
        
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise
