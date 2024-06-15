from sklearn.preprocessing import MinMaxScaler


def normalize(dataframe_2d):
    scaler = MinMaxScaler()
    scaler.fit(dataframe_2d)
    scaled_dataframe = scaler.transform(dataframe_2d)
    return scaled_dataframe


if __name__ == "__main__":
    print(normalize([[1], [2], [3]]))
