# End-to-End Machine Learning Project: Sleep Health and Lifestyle Prediction

![4321660](https://github.com/user-attachments/assets/8c6b8f16-4e1a-48f0-8e49-01f5eaa7ab1f)

## Project Overview

This project aims to build a machine learning model that predicts the likelihood of an individual having a sleep disorder based on various lifestyle and health factors. The dataset includes key features such as sleep duration, physical activity levels, stress, and blood pressure.

The project follows a structured approach from data collection and preprocessing to model building, evaluation, and deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
  - [Data Description](#data-description)
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
  - [Train-Test Split](#train-test-split)
  - [Algorithms Used](#algorithms-used)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Project Workflow

1. **Data Collection:** Using the "Sleep Health and Lifestyle" dataset, which contains information about various health and lifestyle indicators.
2. **Data Preprocessing:** Handling missing values, outliers, and feature scaling.
3. **Exploratory Data Analysis (EDA):** Analyzing the relationships between features and the target variable (Sleep Disorder).
4. **Feature Engineering:** Creating new features or transforming existing features to improve model performance.
5. **Model Building:** Training different machine learning models to predict Sleep Disorders.
6. **Model Evaluation:** Evaluating the performance of models using various metrics.
7. **Model Deployment:** Deploying the final model for real-time predictions.

## Dataset

- **File:** `Sleep_health_and_lifestyle_dataset.csv`
  
### Data Description

The dataset consists of the following columns:
- `Person ID`: Unique identifier for each individual.
- `Gender`: Gender of the individual (Male/Female).
- `Age`: Age of the individual.
- `Occupation`: Occupation category (e.g., Software Engineer, Accountant, Nurse etc).
- `Sleep Duration`: Average sleep duration in hours.
- `Quality of Sleep`: Self-reported sleep quality (scale 1-10).
- `Physical Activity Level`: Daily physical activity level (scale 1-100).
- `Stress Level`: Self-reported stress level (scale 1-10).
- `BMI Category`: Body Mass Index category (Underweight, Normal, Overweight, Obese).
- `Blood Pressure`: Blood pressure reading (e.g., 120/80).
- `Heart Rate`: Average heart rate in beats per minute.
- `Daily Steps`: Average number of steps per day.
- `Sleep Disorder`: Target variable indicating whether an individual has a sleep disorder.i.e Insomnia, Sleep Apnea or No Sleep Disorder.

### Data Cleaning and Preprocessing

- **Missing Values:** Missing values in columns such as `Sleep Disorder` were categorized as 'None' for individuals without disorders.
- **Data Transformation:** Split the `Blood Pressure` column into `Systolic_BP` and `Diastolic_BP`.
- **Scaling:** Numerical features such as `Age`, `Heart Rate`, and `Daily Steps` were standardized using Standard Scalar.
  
## Exploratory Data Analysis (EDA)

EDA was performed using visualizations to understand relationships between lifestyle factors and sleep disorders. Some key findings include:
- **Correlation Analysis:** Identified correlations between physical activity, stress levels, and sleep quality.
- **Visualization:** Used `Plotly` for interactive visualizations and hover effects on various features like Sleep Duration, Quality of Sleep, and BMI.

## Feature Engineering

- **Blood Pressure Split:** Split the `Blood Pressure` column into `Systolic_BP` and `Diastolic_BP` for better feature representation.
- **Categorical Encoding:** Categorical features such as `Gender` and `Occupation` were LabelEncoded for model training.

## Model Building

### Train-Test Split

- The dataset was split into a training set (80%) and a testing set (20%).

### Algorithms Used

- **Decision Tree Classifier**
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **K-Neighbors Classifier**
- **Support Vector Machine (SVM)**

Each model was trained and evaluated using cross-validation.

## Model Evaluation

The performance of the models was evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **R2-Score**

After evaluation, Random Forest emerged as the best-performing model with a high accuracy and balanced precision-recall.

## Model Deployment

The final model was deployed using Flask for a web-based application that allows users to input their lifestyle details and receive predictions on whether they are likely to have a sleep disorder.

### Deployment Steps:
1. Created an API using Flask that accepts input data and returns predictions.
2. The app was containerized using Docker and deployed on AWS ElasticBeanStalk for scalability.

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `pandas` for data manipulation.
  - `scikit-learn` for model building.
  - `seaborn` and `matplotlib` for data visualization.
  - `Flask` for deploying the model.
- **Tools:**
  - `AWS ElasticBeanStalk` for deployment.

## How to Run the Project

To replicate the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/Syed-Irfan-1165/Sleep-Health-and-Lifestyle.git
   cd Sleep-Health-and-Lifestyle

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Train the model:
   ```bash
   python src/components/data_ingestion.py

4. Run the Flask app for model inference:
   ```bash
   python application.py

5. Visit http://localhost:5000 in your browser to use the web app.

## Results
- The Random Forest model achieved an accuracy of 94% on the test data.
- F1 score of 0.91 indicates strong performance in distinguishing between individuals with and without sleep disorders.

## Conclusion and Future Work

This project successfully built a model to predict sleep disorders based on lifestyle data.

## Application Screenshots

### Before Entering Values

<img width="1724" alt="Screenshot 2024-10-06 at 9 33 17 PM" src="https://github.com/user-attachments/assets/763cead4-3faa-4061-8f95-a2401771af59">

<img width="1728" alt="Screenshot 2024-10-06 at 9 33 28 PM" src="https://github.com/user-attachments/assets/687f2527-8989-42cb-8c03-13c0138d8d77">

### After Entering Values

<img width="1728" alt="Screenshot 2024-10-06 at 9 36 14 PM" src="https://github.com/user-attachments/assets/b68507f1-6b74-4289-93c4-6ed52e3e5bb4">

<img width="1728" alt="Screenshot 2024-10-06 at 9 50 54 PM" src="https://github.com/user-attachments/assets/1669aafe-9491-4a40-96e6-06dcf2d048d4">

### Prediction Results

<img width="1728" alt="Screenshot 2024-10-06 at 9 35 12 PM" src="https://github.com/user-attachments/assets/9b6cdd8d-00bf-4aee-a72c-35f1e513d0f8">

