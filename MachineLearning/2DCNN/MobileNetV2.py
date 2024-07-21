import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

# RESOLVE DISCREPANCY
tf.random.set_seed(42)


def MobileNetV2(image_size, num_classes=2):
    # CREATE SEQUENTIAL MODEL
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
    model = tf.keras.Sequential(
        [
            base_model,
            global_average_layer,
            prediction_layer,
        ]
    )

    # MODEL COMPILE
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Plot the validation and training data separately
def PLOT_ACCURACY_LOSS(history, save_path="CNN_Results"):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Plot_Accuracy_Loss.png"))
    plt.show()


def CNN_RESULTS(val_data, best_model, history, class_names, save_path="CNN_Results"):
    # GET VALIDATION DATA FROM GENERATOR
    X_test_batches = []
    y_test_batches = []

    for _ in range(len(val_data)):
        batch = next(val_data)
        X_test_batches.append(batch[0])
        y_test_batches.append(batch[1])

    X_test = np.concatenate(X_test_batches)
    y_test = np.concatenate(y_test_batches)
    y_test = np.argmax(y_test, axis=1)

    # PREDICT TEST DATA
    y_predict = best_model.predict(X_test)
    y_predict = np.argmax(y_predict, axis=1)

    # MODEL EVALUATION
    score = best_model.evaluate(val_data, verbose=2)
    with open(os.path.join(save_path, "Model_Evaluation.txt"), "w") as f:
        f.write("Evaluation Loss: {}\n".format(score[0]))
        f.write("Evaluation Accuracy: {}".format(score[1]))

    # ACCURACY SCORE
    with open(os.path.join(save_path, "Accuracy.txt"), "w") as f:
        print(accuracy_score(y_test, y_predict))
        f.write("Accuracy Score: {}".format(accuracy_score(y_test, y_predict)))

    # PLOT LOSS ACCURACY
    PLOT_ACCURACY_LOSS(history, save_path)

    # PLOT CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_predict, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(save_path, "Plot_Confusion_Matrix.png"))
    plt.show()


def CNN_TRAINING(data_dir, image_size, save_path="CNN_Results"):
    # GET CLASSNAMES
    class_names = [
        x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))
    ]

    # DATA GENERATOR
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # fill_mode="nearest",
    )
    train_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        subset="training",
        class_mode="categorical",
        color_mode="rgb",
    )
    val_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        subset="validation",
        class_mode="categorical",
        color_mode="rgb",
    )

    # CREATE MODEL
    model = MobileNetV2(image_size, num_classes=len(class_names))

    # FIT MODEL
    history = model.fit(
        train_data,
        epochs=100,
        validation_data=val_data,
    )

    # SAVE BEST MODEL
    model.save_weights(os.path.join(save_path, "CNN_Best.weights.h5"))
    model.export(os.path.join(save_path, "CNN_Best"))

    # CONVERT TO TFLITE
    converter = tf.lite.TFLiteConverter.from_saved_model(
        os.path.join(save_path, "CNN_Best")
    )
    tflite_model = converter.convert()
    with open(os.path.join(save_path, "CNN_Best.tflite"), "wb") as f:
        f.write(tflite_model)

    # SAVE HISTORY
    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    # SEE CNN RESULTS
    CNN_RESULTS(val_data, model, history.history, class_names, save_path)


def CNN_EVALUATE(
    data_dir,
    image_size,
    save_path="CNN_Results",
):
    # GET CLASSNAMES
    class_names = [
        x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))
    ]

    # DATA GENERATOR
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2
    )

    val_data = data_gen.flow_from_directory(
        data_dir,
        target_size=image_size,
        subset="validation",
        class_mode="categorical",
        color_mode="rgb",
    )

    # LOAD BEST MODEL
    model = MobileNetV2(image_size, num_classes=len(class_names))
    model.build(input_shape=(None, *(*image_size, 3)))
    model.load_weights(os.path.join(save_path, "CNN_Best.weights.h5"))

    # SEE CNN RESULTS
    history = json.load(open(os.path.join(save_path, "history.json"), "r"))
    CNN_RESULTS(val_data, model, history, class_names, save_path)


def CNN_PREDICT_TF(tflite_path, image_input, image_size):
    img = CNN_PREPROCESS_IMAGE(image_input, image_size)
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
    if abs(result[0][0] - result[0][1]) < 0.5:
        return 2
    result = np.argmax(result, axis=1)
    return result[0]


def CNN_PREDICT(model_path, image_input, image_size, total_class):
    img = CNN_PREPROCESS_IMAGE(image_input, image_size)
    model = MobileNetV2(image_size, num_classes=total_class)
    model.build(input_shape=(None, *(*image_size, 3)))
    model.load_weights(model_path)
    result = model.predict(img)
    result = np.argmax(result, axis=1)
    return result[0]


def CNN_PREPROCESS_IMAGE(image_input, image_size):
    img = cv2.resize(image_input, image_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    data_dir = "SortedFromClient"
    image_size = (224, 224)

    # CNN_TRAINING(data_dir, image_size)

    # CNN_EVALUATE(data_dir, image_size)

    # # TF
    # result = CNN_PREDICT(
    #     os.path.join("CNN_Results", "CNN_Best.weights.h5"),
    #     cv2.imread("1st Dataset/Aircon/20240701063555.png"),
    #     image_size,
    #     5,
    # )
    # print(result)

    # # TFLITE (FASTER)
    # result = CNN_PREDICT_TF(
    #     os.path.join("CNN_Results", "CNN_Best.tflite"),
    #     cv2.imread("1st Dataset/Aircon/20240701063555.png"),
    #     image_size,
    # )
    # print(result)
