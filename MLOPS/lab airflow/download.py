from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import mlflow
from mlflow.models import infer_signature
from train_possum import train_possum_model

# Подавление предупреждений
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def process_data():
    # Загрузка данных
    df = pd.read_csv('https://raw.githubusercontent.com/azzrdn/Possum-Regression-Project/1fb5b851346ca03c909fb897b5bcd4a24a902a74/possum.csv')
    print(df)

    # Предварительная обработка
    df = df.dropna(subset=['sex'])  # Удаляем строки с NaN в целевом признаке
    df['sex'] = df['sex'].map({'m': 0, 'f': 1})  # Конвертируем в числовой формат
    
    df.to_csv('df_clear.csv')
    print("Данные успешно обработаны и сохранены в df_clear.csv")

dag_possum = DAG(
    dag_id="train_possum",
    start_date=datetime(2025,1,4),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
process_task = PythonOperator(python_callable=process_data, task_id = "download_data", dag=dag_possum)
train_task = PythonOperator(python_callable=train_possum_model, task_id = "train_possum", dag = dag_possum)
process_task >> train_task

