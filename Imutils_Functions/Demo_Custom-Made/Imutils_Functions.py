import imutils
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("demo_images/1.jpg")

# Translation
translated = imutils.translate(image, 10, 10)
cv2.imshow("Translated", translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotation
rotated = imutils.rotate(image, angle=90)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize
resized = imutils.resize(image, width=500)
cv2.imshow("Resize.With.Respect.To.AspectRatio", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Skeletonization
image_to_skeleton = cv2.imread("demo_images/pyima.jpg")
gray = cv2.cvtColor(image_to_skeleton, cv2.COLOR_BGR2GRAY)
skeleton = imutils.skeletonize(gray, size=(3, 3))
cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Automatic Canny Edge Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edgeMap = imutils.auto_canny(gray)
cv2.imshow("Auto-Canny_Original", image)
cv2.imshow("Auto-Canny_Final", edgeMap)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying with Matplotlib
plt.figure("Correct")
plt.imshow(imutils.opencv2matplotlib(image))
plt.show()
