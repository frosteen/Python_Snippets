import cv2

cap = cv2.VideoCapture(0)

cv2.namedWindow("frame")

# Create Trackbar
cv2.createTrackbar("x", "frame", 10, 255, lambda x: x)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If there is no frame then break
    if not ret:
        break

    # Get trackbar value
    print(cv2.getTrackbarPos("x", "frame"))

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # ESC to quit
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
