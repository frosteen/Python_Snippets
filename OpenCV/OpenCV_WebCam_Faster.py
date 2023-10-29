import cv2
from imutils.video import VideoStream

# Using VideoStream from imutils (fast because of threading)
cap = VideoStream(0).start()

while True:
    # Capture frame-by-frame
    frame = cap.read()

    if not cap.grabbed:
        continue

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # ESC to quit
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

# When everything done, release the capture
cap.stop()
cv2.destroyAllWindows()
