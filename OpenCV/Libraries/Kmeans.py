import cv2
import matplotlib.pyplot as plt
import numpy as np


def graph_kmeans(title, data, label, center):
    # Graph Kmeans
    label = label.flatten()
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(
            data[label == i, 0],
            data[label == i, 1],
            label=str(center[i]),
            color=(np.float32(np.flip(center, 1)) / 255)[i],
        )
    plt.scatter(center[:, 0], center[:, 1], s=80, color="k")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def color_quantization(image, clusters=3):
    """Performs color quantization using K-means clustering algorithm"""

    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(
        data, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # At this point we can make the image with k colors
    # Convert center to uint8:
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result, data, label, center
