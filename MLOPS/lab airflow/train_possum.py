import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import mlflow
from mlflow.models import infer_signature
import joblib  # Добавлен импорт для joblib


def train_possum_model():
    """Функция для обучения модели Possum, которую можно использовать в Airflow DAG"""
    df = pd.read_csv("./df_clear.csv")
    X = df.drop("sex", axis=1)
    y = df["sex"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Преобразование данных (Imputer для числовых признаков)
    numeric_transformer = SimpleImputer(strategy='median')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X_train.select_dtypes(include=['int64', 'float64']).columns)
        ])

    # ==============================================
    # Выберите модель, раскомментировав нужную часть
    # ==============================================

    # Вариант 1: SGDClassifier (раскомментируйте для использования)
    pipeline = make_pipeline(
        preprocessor,
        SGDClassifier(random_state=42)
    )

    params = {
        'sgdclassifier__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'sgdclassifier__l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'sgdclassifier__penalty': ['l1', 'l2', 'elasticnet'],
        'sgdclassifier__eta0': np.linspace(0.1, 1, 4),
    }

    mlflow.set_experiment("possum_sex_classification_sgd")

    # Вариант 2: RandomForestClassifier (раскомментируйте для использования)
    '''
    pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=42)
    )

    params = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [3, 5, 10, None],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
    }

    mlflow.set_experiment("possum_sex_classification_rf")
    '''
    '''
    # Вариант 3: LogisticRegression (раскомментируйте для использования)
    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(random_state=42)
    )

    params = {
        'logisticregression__C': [0.01, 0.1, 1, 10, 100],
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__solver': ['liblinear', 'saga']
    }

    mlflow.set_experiment("possum_sex_classification_logreg")
    '''
    # ==============================================
    # Общая часть для любой модели
    # ==============================================
    with mlflow.start_run():
        clf = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)
        
        # Метрики
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        
        # Логирование
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        # Сохранение модели в файл
        with open("best_model.pkl", "wb") as file:
            joblib.dump(best_model, file)
        
        print(f"Лучшая accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    # Поиск лучшей модели среди всех экспериментов
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print("Путь к лучшей модели:", path2model)

