# Standard Python
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import cv2


class GMM_Image_Segmentation:
    def __init__(self, image, scale, n_components=4):
        H, W = image.shape[:2]
        image = cv2.resize(
            image, (int(H * scale), int(W * scale)), interpolation=cv2.INTER_AREA
        )
        image_reshape = image.reshape((-1, 3))  # convert to 2D
        gmm_model = GMM(
            n_components=n_components,
            covariance_type="tied",
        ).fit(image_reshape)
        gmm_labels = gmm_model.predict(image_reshape)
        SCALED_H, SCALED_W = image.shape[:2]
        self.segmented_image = gmm_labels.reshape(SCALED_H, SCALED_W)
        self.segmented_image = self.segmented_image.astype(np.uint8)
        self.segmented_image = cv2.resize(
            self.segmented_image, (SCALED_H, SCALED_W), interpolation=cv2.INTER_LINEAR
        )

    def adjusted_segmented_image(self, contrast=1, brightness=0):
        image = cv2.addWeighted(
            self.segmented_image,
            contrast,
            self.segmented_image,
            0,
            brightness,
        )
        return image


def nothing(x):
    pass


def initialize_trackbars(
    min=-255, max=255, intialTracbarVals=0, windowName="Trackbars"
):
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 360, 240)
    cv2.createTrackbar("Threshold1", windowName, intialTracbarVals, 255, nothing)
    cv2.setTrackbarMin("Threshold1", windowName, min)
    cv2.setTrackbarMax("Threshold1", windowName, max)
    cv2.createTrackbar("Threshold2", windowName, intialTracbarVals, 255, nothing)
    cv2.setTrackbarMin("Threshold2", windowName, min)
    cv2.setTrackbarMax("Threshold2", windowName, max)


def get_val_tackbars(windowName="Trackbars"):
    Threshold1 = cv2.getTrackbarPos("Threshold1", windowName)
    Threshold2 = cv2.getTrackbarPos("Threshold2", windowName)
    src = Threshold1, Threshold2
    return src


if __name__ == "__main__":
    initialize_trackbars(-255, 255, 100)
    img = cv2.imread("Test_Pictures/1000PHP.jpg")
    H, W = img.shape[:2]
    GMM_Image_Segmentation = GMM_Image_Segmentation(img, 1, 4)
    while True:
        thres1, thres2 = get_val_tackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
        img = GMM_Image_Segmentation.adjusted_segmented_image(127, 0)
        # img = GMM_Image_Segmentation.adjusted_segmented_image(thres1, thres2)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", int(W / 2), int(H / 2))
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
