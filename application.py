from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

sleep_disorder_labels = {0: 'Insomnia', 1: 'No Sleep Disorder', 2: 'Sleep Apnea'}

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Gender=request.form.get('gender'),  # Changed to lowercase
            Occupation=request.form.get('occupation'),
            BMI_Category=request.form.get('bmi_category'),
            Age=request.form.get('age'),
            Sleep_Duration=request.form.get('sleep_duration'),
            Quality_of_Sleep=request.form.get('quality_of_sleep'),
            Physical_Activity_Level=request.form.get('physical_activity_level'),
            Stress_Level=request.form.get('stress_level'),
            Heart_Rate=request.form.get('heart_rate'),
            Daily_Steps=request.form.get('daily_steps'),
            High_BP=request.form.get('high_bp'),
            Low_BP=request.form.get('low_bp')
        )


        pred_df=data.get_data_as_data_frame()
        pred_df.columns = ['Gender', 'Occupation', 'BMI Category', 'Age', 'Sleep Duration',
                           'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
                           'Heart Rate', 'Daily Steps', 'High_BP', 'Low_BP']
        # Check the column names in your dataframe (for debugging)
        print("Actual column names:", pred_df.columns)
        # print(pred_df)
        
        predict_pipeline=PredictPipeline()
        try:
            predict_pipeline = predict_pipeline.predict(pred_df)
            results = sleep_disorder_labels.get(predict_pipeline[0], "Unknown Disorder")
            return render_template('home.html', results=results)
        except Exception as e:
            # Handle and print any errors that occur during prediction
            print(f"Error during prediction: {str(e)}")
            return render_template('home.html', error="An error occurred during prediction. Please try again.")

        # results=predict_pipeline.predict(pred_df)
        # return render_template('home.html',results=results[0])
    
if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)