import os

import pandas as pd
from joblib import dump, load
from neupy import algorithms
from sklearn.metrics import accuracy_score


def Train_LVQ(df_train_x, df_train_y, df_test_x, df_test_y, save_path=""):
    # dynamic inputs and classes
    n_inputs = len(df_train_x.columns)
    n_classes = len(pd.unique(df_train_y))

    # LVQ Network
    lvqnet = algorithms.LVQ(n_inputs=n_inputs, n_classes=n_classes, verbose=True)
    lvqnet.train(df_train_x, df_train_y, epochs=200)
    lvqnet.plot_errors()

    # Save pre-trained LVQ network
    dump(lvqnet, os.path.join(save_path, "LVQ.joblib"))

    # Predict
    predict_classes = lvqnet.predict(df_test_x)

    # Accuracy
    accuracy = accuracy_score(df_test_y, predict_classes)

    return predict_classes, accuracy


def Process_LVQ(df_test_x, lvq_joblib_path=""):
    # Load LVQ network from the joblib file
    lvqnet = load(lvq_joblib_path)

    # predict
    predict_classes = lvqnet.predict(df_test_x)

    return predict_classes
