import sys, os
import numpy as np
import pandas as pd
from src.exception import  CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    

    def predict(self,features):
        try:
            model_path=os.path.join("/Users/syed/Documents/Sleep-Health-and-Lifestyle/artifacts/model.pkl")
            preprocessor_path=os.path.join('/Users/syed/Documents/Sleep-Health-and-Lifestyle/artifacts/preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                 Gender: str,
                 Occupation: str,
                 BMI_Category: str,
                 Age:int,
                 Sleep_Duration: float,
                 Quality_of_Sleep: int,
                 Physical_Activity_Level: int,
                 Stress_Level: int,
                 Heart_Rate: int,
                 Daily_Steps:int,
                 High_BP:float,
                 Low_BP:float):

                self.Gender = Gender
                
                self.Occupation = Occupation
                
                self.BMI_Category = BMI_Category
                
                self.Age = Age
                
                self.Sleep_Duration = Sleep_Duration
                
                self.Quality_of_Sleep = Quality_of_Sleep
                
                self.Physical_Activity_Level = Physical_Activity_Level
                
                self.Stress_Level = Stress_Level
                
                self.Heart_Rate = Heart_Rate
                
                self.Daily_Steps = Daily_Steps
                
                self.High_BP = High_BP
                
                self.Low_BP = Low_BP
                
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict ={
                "Gender": [self.Gender],
                "Occupation":[self.Occupation],
                "BMI_Category":[self.BMI_Category ],
                "Age":[self.Age],
                "Sleep_Duration":[self.Sleep_Duration],
                "Quality_of_Sleep":[self.Quality_of_Sleep],
                "Physical_Activity_Level":[self.Physical_Activity_Level],
                "Stress_Level":[self.Stress_Level],
                "Heart_Rate":[self.Heart_Rate],
                "Daily_Steps":[self.Daily_Steps],
                "High_BP":[self.High_BP],
                "Low_BP":[self.Low_BP],
                
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)