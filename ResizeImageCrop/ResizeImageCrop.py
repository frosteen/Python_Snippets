import cv2
import numpy as np


def resize_image_crop(img, size=(1280, 1280)):
    """
    This crops first the center image as a square then resize
    """
    # crop_square
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    resized = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)

    return resized


if __name__ == "__main__":
    img = cv2.imread(
        "C:\\Users\\Luis Daniel Pambid\\Pictures\\Screenshots\\Screenshot (560).png"
    )
    img = resize_image(img, (256, 256))
    cv2.namedWindow("IMG", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IMG", (500, 500))
    cv2.imshow("IMG", img)
    cv2.waitKey(0)
