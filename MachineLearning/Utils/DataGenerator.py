import os

import numpy as np
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        X_set_path,
        y_set_path,
        batch_size=32,
        shuffle=True,
    ):
        "Initialization"
        self.X_set_path = X_set_path
        self.id_list = np.array(
            [
                int(os.path.splitext(filename)[0])
                for filename in os.listdir(self.X_set_path)
            ]
        )  # get only the filename without extension
        self.y_set_path = y_set_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate batch of data for sequences
        sequences = np.array(
            [
                np.load(
                    os.path.join(
                        self.X_set_path,
                        "{}.npy".format(index),
                    )
                )
                for index in indexes
            ]
        )

        # Generate batch of data for labels
        labels = np.array(
            [
                np.load(
                    os.path.join(
                        self.y_set_path,
                        "{}.npy".format(index),
                    )
                )
                for index in indexes
            ]
        )

        return sequences, labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.sort(self.id_list)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# FOR TESTING PURPOSES
if __name__ == "__main__":
    batch_size = 16
    training_generator = DataGenerator(
        os.path.join("ASL_PREPROCESS", "DATA", "TRAINING"),
        os.path.join("ASL_PREPROCESS", "LABELS"),
        batch_size=batch_size,
    )

    validation_generator = DataGenerator(
        os.path.join("ASL_PREPROCESS", "DATA", "VALIDATION"),
        os.path.join("ASL_PREPROCESS", "LABELS"),
        batch_size=batch_size,
    )

    testing_generator = DataGenerator(
        os.path.join("ASL_PREPROCESS", "DATA", "TESTING"),
        os.path.join("ASL_PREPROCESS", "LABELS"),
        batch_size=batch_size,
    )

    print("testing")
    for x in testing_generator:
        print(x[0].shape, x[1])

    print("validation")
    for x in validation_generator:
        print(x[0].shape, x[1])

    print("training")
    for x in training_generator:
        print(x[0].shape, x[1])
