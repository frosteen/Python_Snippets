import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import Input, Sequential, losses, metrics, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PlotConfusionMatrix import plot_confusion_matrix

# RESOLVE DISCREPANCY
tf.keras.backend.set_learning_phase(0)
tf.random.set_seed(42)


def CNN_Model(image_size):
    # CREATE SEQUENTIAL MODEL
    model = Sequential(
        [
            Input(shape=(image_size[0], image_size[1], 3)),
            Conv2D(filters=10, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            # Dense(units=256, activation="relu"),
            # Dropout(0.5),
            Dense(units=1, activation="sigmoid"),
        ]
    )

    # MODEL COMPILE
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

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
    data_gen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)

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

    # FIT MODEL
    history = model.fit(
        train_data,
        epochs=50,
        validation_data=val_data,
    )

    # SAVE MODEL
    model.save_weights(os.path.join(save_path, "CNN.keras"))

    # Convert to TF Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save TF Lite Model
    tf.io.write_file(os.path.join(save_path, "CNN.tflite"), tflite_model)

    # PLOT LOSS ACCURACY
    plot_loss_curves(history, save_path)

    # GET TEST DATA FROM GENERATOR
    X_test = np.concatenate([val_data.next()[0] for _ in range(val_data.__len__())])
    y_test = np.concatenate([val_data.next()[1] for _ in range(val_data.__len__())])

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


def CNN_Evaluate(weights_path, data_dir, image_size, save_path="CNN_Results"):
    # DATA GENERATOR
    data_gen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)

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
    X_test = np.concatenate([val_data.next()[0] for _ in range(val_data.__len__())])
    y_test = np.concatenate([val_data.next()[1] for _ in range(val_data.__len__())])

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
    data_dir = r"C:\Users\Luis Daniel Pambid\Downloads\archive\pizza_not_pizza"

    # CNN_Training(data_dir, (32, 32))

    # CNN_Evaluate(os.path.join("CNN_Results", "CNN.keras"), data_dir, (32, 32))

    # TF
    # image = cv2.imread("Test_Pictures/1000PHP.jpg")
    # result = CNN_Predict(os.path.join("CNN_Results", "CNN.keras"), image, (32, 32))
    # print(result)

    # TF LITE (FASTER)
    image = cv2.imread("Test_Pictures/1000PHP.jpg")
    result = CNN_Predict_TF(os.path.join("CNN_Results", "CNN.tflite"), image, (32, 32))
    print(result)
