from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class Detector:
    def __init__(self, model_type="OD", threshold=0.5, use_gpu=True):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type == "OD":  # object detection
            yaml_file = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            self.cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

        elif model_type == "IS":  # instance segmentation
            yaml_file = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            self.cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

        elif model_type == "KP":  # keypoint detection
            yaml_file = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
            self.cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

        elif model_type == "LVIS":  # LVIS segmentation
            yaml_file = (
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
            )
            self.cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

        elif model_type == "PS":  # panoptic segmentation
            yaml_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
            self.cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

        # set threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # gpu or cpu
        if use_gpu:
            self.cfg.MODEL.DEVICE = "cuda"
        else:
            self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def on_image(self, image):
        if self.model_type != "PS":
            predictions = self.predictor(image)["instances"]
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            )
            output = viz.draw_instance_predictions(predictions.to("cpu"))

            pred_classes = predictions.pred_classes.cpu().tolist()
            pred_boxes = predictions.pred_boxes.tensor.cpu().tolist()
            class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
            pred_class_names = list(map(lambda x: class_names[x], pred_classes))

            class_names_boxes = []
            for class_name, box in zip(pred_class_names, pred_boxes):
                class_names_boxes.append({"class_name": class_name, "boxes": box})

            return output.get_image()[:, :, ::-1], class_names_boxes
        else:
            predictions, segment_info = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            )
            output = viz.draw_panoptic_seg_predictions(
                predictions.to("cpu"), segment_info
            )

            # semantic classes
            semantic_pred_classes = list(
                item["category_id"] for item in segment_info if "score" not in item
            )
            semantic_class_names = MetadataCatalog.get(
                self.cfg.DATASETS.TRAIN[0]
            ).stuff_classes
            semantic_pred_class_names = list(
                map(lambda x: semantic_class_names[x], semantic_pred_classes)
            )

            return output.get_image()[:, :, ::-1], semantic_pred_class_names
