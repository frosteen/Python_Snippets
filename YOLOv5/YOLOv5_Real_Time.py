import cv2
import torch

model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
)
model.to("cuda")
model.eval()
model.conf = 0.5

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        results = model(frame)
    results_df = results.pandas().xyxy[0]
    results.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    print(results_df)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
