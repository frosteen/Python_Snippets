import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from RetinaNet.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# define the torchvision image transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # add a batch dimension
    with torch.no_grad():
        outputs = model(image)  # get the predictions on the image
    # get all the scores
    scores = list(outputs[0]["scores"].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [
        scores.index(i) for i in scores if i > detection_threshold
    ]
    # get all the predicted bounding boxes
    bboxes = outputs[0]["boxes"].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
    # get all the predicited class names
    labels = outputs[0]["labels"].cpu().numpy()
    pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes


def draw_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[coco_names.index(classes[i])]
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
        )
        cv2.putText(
            image,
            classes[i],
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return image
