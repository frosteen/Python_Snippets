import cv2
import numpy as np


def cam_scanner(
    img,
    w_img_warp,
    h_img_warp,
    canny_edge_thres1=200,
    canny_edge_thres2=200,
    strip_pixels_side=10,
    interpolation=cv2.INTER_LINEAR,  # or cv2.INTER_CUBIC, GOOD FOR ENLARGING
):
    """
    Just like cam scanner :)
    """

    ## IMAGE PROCESSING / APPLYING FILTERS
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    # img_denoise = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)  # ADD DENOISING
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_canny = cv2.Canny(
        img_blur, canny_edge_thres1, canny_edge_thres2
    )  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)  # APPLY DILATION
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    img_contours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, _ = cv2.findContours(
        img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # FIND ALL CONTOURS
    cv2.drawContours(
        img_contours, contours, -1, (0, 255, 0), 5
    )  # DRAW ALL DETECTED CONTOURS

    ## FIND THE BIGGEST COUNTOUR
    biggest, _ = biggest_contour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32(
            [[0, 0], [w_img_warp, 0], [0, h_img_warp], [w_img_warp, h_img_warp]]
        )  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp_colored = cv2.warpPerspective(img, matrix, (w_img_warp, h_img_warp))

        p = strip_pixels_side
        img_warp_colored = img_warp_colored[
            p : img_warp_colored.shape[0] - p, p : img_warp_colored.shape[1] - p
        ]  # REMOVE p PIXELS FORM EACH SIDE
        img_warp_colored = cv2.resize(
            img_warp_colored, (w_img_warp, h_img_warp), interpolation=interpolation
        )

        return img_warp_colored

    return img_contours


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def nothing(x):
    pass


def initialize_trackbars(intialTracbarVals=0, windowName="Trackbars"):
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 360, 240)
    cv2.createTrackbar("Threshold1", windowName, intialTracbarVals, 255, nothing)
    cv2.createTrackbar("Threshold2", windowName, intialTracbarVals, 255, nothing)


def get_val_tackbars(windowName="Trackbars"):
    Threshold1 = cv2.getTrackbarPos("Threshold1", windowName)
    Threshold2 = cv2.getTrackbarPos("Threshold2", windowName)
    src = Threshold1, Threshold2
    return src


if __name__ == "__main__":
    X, Y = 160, 66
    img = cv2.imread("Test_Pictures/1000PHP.jpg")
    H, W = img.shape[:2]
    initialize_trackbars(0)
    while True:
        thres1, thres2 = get_val_tackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
        scan = cam_scanner(img, int(W), int(Y * W / X), thres1, thres2, 10)
        h, w = scan.shape[:2]
        scale = 1
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", int(w * scale), int(h * scale))
        cv2.imshow("Result", scan)
        if cv2.waitKey(1) & 0xFF == 27:
            break
