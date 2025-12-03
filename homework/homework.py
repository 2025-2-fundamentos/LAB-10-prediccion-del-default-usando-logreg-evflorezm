# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# flake8: noqa: E501
# pylint: disable=import-outside-toplevel, line-too-long

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import pickle
import gzip
import os
import json


# -------------------------------------------------------
# Cargar datos desde ZIP
# -------------------------------------------------------
def load_data(csv_file):
    return pd.read_csv(csv_file, compression="zip")


# -------------------------------------------------------
# Paso 1 – Limpieza del dataset
# -------------------------------------------------------
def data_clean(df):
    df = df.copy()

    # Renombrar columna objetivo
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Remover ID
    if "ID" in df.columns:
        df.drop(columns="ID", inplace=True)

    # Remover registros inválidos
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    # EDUCATION > 4 -> "others" = 4
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


# -------------------------------------------------------
# Paso 2 – Dividir train/test
# -------------------------------------------------------
def split_data(df_train, df_test):
    x_train = df_train.drop(columns="default")
    y_train = df_train["default"]

    x_test = df_test.drop(columns="default")
    y_test = df_test["default"]

    return x_train, y_train, x_test, y_test


# -------------------------------------------------------
# Paso 3 – Pipeline
# -------------------------------------------------------
def create_pipeline():

    categorical_cols = ["EDUCATION", "SEX", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder=MinMaxScaler()  # escala todo lo numérico
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression, k=10)),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    return pipeline


# -------------------------------------------------------
# Paso 4 – GridSearchCV
# -------------------------------------------------------
def make_grid_search(pipeline):

    param_grid = {
        "feature_selection__k": list(range(1, 11)),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__max_iter": [100, 200, 400]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=0
    )

    return grid


# -------------------------------------------------------
# Paso 5 – Guardar modelo
# -------------------------------------------------------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


# -------------------------------------------------------
# Paso 6 – Métricas
# -------------------------------------------------------
def compute_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred))
    }, y_pred


# -------------------------------------------------------
# Paso 7 – Matriz de confusión
# -------------------------------------------------------
def compute_cm(y_true, y_pred, dataset_name):

    cm = confusion_matrix(y_true, y_pred)

    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1])
        }
    }


# -------------------------------------------------------
# MAIN – Ejecuta todo
# -------------------------------------------------------
def main():
    # Crear carpeta output si no existe
    os.makedirs("files/output", exist_ok=True)

    # Cargar datos
    df_train = data_clean(load_data("files/input/train_data.csv.zip"))
    df_test = data_clean(load_data("files/input/test_data.csv.zip"))

    # Split
    x_train, y_train, x_test, y_test = split_data(df_train, df_test)

    # Pipeline
    pipeline = create_pipeline()

    # GridSearch
    grid = make_grid_search(pipeline)
    model = grid.fit(x_train, y_train)

    # Métricas train/test
    metrics_train, y_pred_train = compute_metrics(model, x_train, y_train, "train")
    metrics_test, y_pred_test = compute_metrics(model, x_test, y_test, "test")

    # Matrices de confusión
    cm_train = compute_cm(y_train, y_pred_train, "train")
    cm_test = compute_cm(y_test, y_pred_test, "test")

    # Guardar metrics.json
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")
        f.write(json.dumps(cm_train) + "\n")
        f.write(json.dumps(cm_test) + "\n")

    # Guardar modelo
    save_model(model, "files/models/model.pkl.gz")


if __name__ == "__main__":
    main()
