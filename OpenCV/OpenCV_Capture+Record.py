import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)

is_video_record = False

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If there is no frame then break
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break
    elif key & 0xFF == ord("r"):
        dt_string = datetime.now().strftime("%d.%m.%Y %H_%M_%S")
        record_path = f"Record-{dt_string}.avi"
        print(f"record: {record_path}")
        h, w, _c = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(record_path, fourcc, 10.0, (w, h))
        is_video_record = True
    elif key & 0xFF == ord("f"):
        print(f"record stopped. {record_path}")
        video_writer.release()
        is_video_record = False
    elif key & 0xFF == ord("c"):
        dt_string = datetime.now().strftime("%d.%m.%Y %H_%M_%S")
        image_path = f"Captured-{dt_string}.jpg"
        print(f"captured: {image_path}")
        cv2.imwrite(image_path, frame)

    if is_video_record:
        print("recording")
        video_writer.write(frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
