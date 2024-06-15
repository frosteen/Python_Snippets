import os

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def Train_PCA(
    df_train_x,
    df_train_y,
    new_columns=["PC1", "PC2"],
    dimension=2,
    save_path="",
):
    # get x values
    x = df_train_x.values

    # normalizer
    normalizer = MinMaxScaler()
    x_normalize = normalizer.fit_transform(x)
    dump(normalizer, os.path.join(save_path, "Normalizer.joblib"))

    # PCA
    pca = PCA(n_components=dimension)
    principal_components = pca.fit_transform(x_normalize)
    dump(pca, os.path.join(save_path, "PCA.joblib"))

    # combine principal_components and classes
    principal_df = pd.DataFrame(data=principal_components, columns=new_columns)
    df_train_y.reset_index(drop=True, inplace=True)
    final_df = pd.concat([principal_df, df_train_y], axis=1)

    return final_df


def Process_PCA(
    df_test_x,
    df_test_y,
    new_columns=["PC1", "PC2"],
    normalizer_joblib_path="",
    pca_joblib_path="",
):
    # get x values
    x = df_test_x.values

    # normalizer
    normalizer = load(normalizer_joblib_path)
    x_normalize = normalizer.transform(x)

    # PCA
    pca = load(pca_joblib_path)
    principal_components = pca.transform(x_normalize)

    # combine principal_components and classes
    principal_df = pd.DataFrame(data=principal_components, columns=new_columns)
    df_test_y.reset_index(drop=True, inplace=True)
    final_df = pd.concat([principal_df, df_test_y], axis=1)

    return final_df


def Graph_2DPCA(df, title="PCA"):
    # get unique classes
    targets = df.iloc[:, -1].unique()
    for target in targets:
        indicesToKeep = df.iloc[:, -1] == target
        plt.scatter(
            df.loc[indicesToKeep, df.columns[0]],
            df.loc[indicesToKeep, df.columns[1]],
            s=50,
        )

    # plot
    plt.xlabel(df.columns[0], fontsize=15)
    plt.ylabel(df.columns[1], fontsize=15)
    plt.title(title, fontsize=20)
    plt.legend(targets, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid()
    plt.show()
