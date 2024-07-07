import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


def CNN_PREDICT_TF(tflite_path, image_input, image_size):
    img = CNN_PREPROCESS_IMAGE(image_input, image_size)
    img = img.astype(np.float32)

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_path)
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


def CNN_PREPROCESS_IMAGE(image_input, image_size):
    img = cv2.resize(image_input, image_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img
