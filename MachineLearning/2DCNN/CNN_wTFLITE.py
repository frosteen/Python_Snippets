import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import Input, Sequential, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from PlotConfusionMatrix import plot_confusion_matrix

# RESOLVE DISCREPANCY
tf.keras.backend.set_learning_phase(0)
tf.random.set_seed(42)


def CNN_Model(image_size):
    # CREATE SEQUENTIAL MODEL
    model = Sequential(
        [
            Input(shape=(image_size[0], image_size[1], 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=256, activation="relu"),
            Dropout(0.5),
            Dense(units=1, activation="sigmoid"),
        ]
    )

    # MODEL COMPILE
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # PRINT MODEL SUMMARY
    model.summary()

    return model


# Plot the validation and training data separately
def plot_loss_curves(history, save_path=""):
    """
    Plots the curves of both loss and accuracy
    """

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Plot_Accuracy_Loss.png"))
    plt.show()


def CNN_Training(data_dir, image_size, save_path="CNN_Results"):
    # DATA GENERATOR
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=32,
        subset="training",
        class_mode="binary",
        color_mode="rgb",
    )

    val_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=32,
        subset="validation",
        class_mode="binary",
        color_mode="rgb",
    )

    # CREATE MODEL
    model = CNN_Model(image_size)

    # GET BEST MODEL CHECKPOINT
    checkpoint_path = os.path.join(save_path, "CNN_Best.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    # FIT MODEL
    history = model.fit(
        train_data,
        epochs=20,
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        callbacks=[checkpoint],
    )

    # SAVE MODEL
    model.save_weights(os.path.join(save_path, "CNN_Last.keras"))

    # Convert to TF Lite format
    best_model = CNN_Model(image_size)
    best_model.load_weights(os.path.join(save_path, "CNN_Best.keras"))
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()

    # Save TF Lite Model
    tf.io.write_file(os.path.join(save_path, "CNN_Best.tflite"), tflite_model)

    # PLOT LOSS ACCURACY
    plot_loss_curves(history, save_path)

    # GET TEST DATA FROM GENERATOR
    X_test_batches = []
    y_test_batches = []

    for _ in range(val_data.__len__()):
        batch = val_data.next()
        X_test_batches.append(batch[0])
        y_test_batches.append(batch[1])

    X_test = np.concatenate(X_test_batches)
    y_test = np.concatenate(y_test_batches)

    # PREDICT TEST DATA
    y_predict = best_model.predict(X_test)
    y_predict = [0 if val < 0.5 else 1 for val in y_predict]

    # MODEL EVALUATION
    score = best_model.evaluate(val_data, verbose=2)
    with open(os.path.join(save_path, "Model_Evaluation.txt"), "w") as f:
        f.write("Evaluation Loss: {}\n".format(score[0]))
        f.write("Evaluation Accuracy: {}".format(score[1]))

    # ACCURACY SCORE
    with open(os.path.join(save_path, "Accuracy.txt"), "w") as f:
        print(accuracy_score(y_test, y_predict))
        f.write("Accuracy Score: {}".format(accuracy_score(y_test, y_predict)))

    # CONFUSION MATRIX
    confusion_matrix_result = confusion_matrix(
        y_test, y_predict, labels=np.unique(y_test)
    )
    plot_confusion_matrix(
        confusion_matrix_result,
        os.listdir(data_dir),
        title="Plot_Confusion_Matrix",
        save_path=save_path,
        normalize=False,
    )


def CNN_Evaluate(weights_path, data_dir, image_size, save_path="CNN_Results"):
    # DATA GENERATOR
    data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    val_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=32,
        subset="validation",
        class_mode="binary",
        color_mode="rgb",
    )

    # CREATE MODEL
    model = CNN_Model(image_size)

    # LOAD MODEL
    model.load_weights(weights_path)

    # GET TEST DATA FROM GENERATOR
    X_test_batches = []
    y_test_batches = []

    for _ in range(val_data.__len__()):
        batch = val_data.next()
        X_test_batches.append(batch[0])
        y_test_batches.append(batch[1])

    X_test = np.concatenate(X_test_batches)
    y_test = np.concatenate(y_test_batches)

    # PREDICT TEST DATA
    y_predict = model.predict(X_test)
    y_predict = [0 if val < 0.5 else 1 for val in y_predict]

    # MODEL EVALUATION
    score = model.evaluate(val_data, verbose=2)
    with open(os.path.join(save_path, "Model_Evaluation.txt"), "w") as f:
        f.write("Evaluation Loss: {}\n".format(score[0]))
        f.write("Evaluation Accuracy: {}".format(score[1]))

    # ACCURACY SCORE
    with open(os.path.join(save_path, "Accuracy.txt"), "w") as f:
        print(accuracy_score(y_test, y_predict))
        f.write("Accuracy Score: {}".format(accuracy_score(y_test, y_predict)))

    # CONFUSION MATRIX
    confusion_matrix_result = confusion_matrix(
        y_test, y_predict, labels=np.unique(y_test)
    )
    plot_confusion_matrix(
        confusion_matrix_result,
        os.listdir(data_dir),
        title="Plot_Confusion_Matrix",
        save_path=save_path,
        normalize=False,
    )


def CNN_Predict_TF(tflite_path, image_input, image_size):
    img = CNN_Preprocess_Image(image_input, image_size)
    img = img.astype(np.float32)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    interpreter.set_tensor(input_details[0]["index"], img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    result = interpreter.get_tensor(output_details[0]["index"])

    result = [0 if val < 0.5 else 1 for val in result]
    return result[0]


def CNN_Predict(weights_path, image_input, image_size):
    img = CNN_Preprocess_Image(image_input, image_size)
    model = CNN_Model(image_size)
    model.load_weights(weights_path)
    result = model.predict(img)
    result = [0 if val < 0.5 else 1 for val in result]
    return result[0]


def CNN_Preprocess_Image(image_input, image_size):
    img = cv2.resize(image_input, image_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    data_dir = r"DATASET"

    CNN_Training(data_dir, (64, 64))

    # CNN_Evaluate(os.path.join("CNN_Results", "CNN_Best.keras"), data_dir, (64, 64))

    # # TF
    # image = cv2.imread("DATASET/FERTILE/20240213_194825.jpg")
    # result = CNN_Predict(os.path.join("CNN_Results", "CNN_Best.keras"), image, (64, 64))
    # print(result)

    # # TF LITE (FASTER)
    # image = cv2.imread("DATASET/INFERTILE/20240225_054302.jpg")
    # result = CNN_Predict_TF(
    #     os.path.join("CNN_Results", "CNN_Best.tflite"), image, (64, 64)
    # )
    # print(result)
