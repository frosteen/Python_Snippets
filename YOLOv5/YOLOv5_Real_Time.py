import cv2
import torch

model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
)
model.cuda()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_to_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # Must convert to RGB before inputting to the model
    results = model(frame_to_rgb)
    results_df = results.pandas().xyxy[0]
    results_to_bgr = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)

    print(results_df)

    cv2.imshow("frame", results_to_bgr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
