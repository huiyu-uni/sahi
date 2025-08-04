
import sys 
import os 
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
import cv2

from sahi.slicing import slice_image
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.utils.cv import visualize_object_predictions


from typing import Generator, List, Optional, Union
import time

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

def filter_predictions(object_prediction_list, exclude_classes_by_name, exclude_classes_by_id):
    return [
        obj_pred
        for obj_pred in object_prediction_list
        if obj_pred.category.name not in (exclude_classes_by_name or [])
        and obj_pred.category.id not in (exclude_classes_by_id or [])
    ]

def create_object_prediction_list_from_original_predictions(
    original_predictions,
    shift_amount: Optional[List[int]] = [0, 0],
    full_shape: Optional[List[int]] = None,
):
    # handle all predictions
    object_prediction_list_per_image = []
    # for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
    for image_ind, xyxy_conf_cls in enumerate(original_predictions):
        object_prediction_list = []

        # process predictions
        for prediction in xyxy_conf_cls.cpu().detach().numpy():
            
            x1 = prediction[0]
            y1 = prediction[1]
            x2 = prediction[2]
            y2 = prediction[3]
            bbox = [x1, y1, x2, y2]
            
            score = prediction[4]
            category_id = int(prediction[5])
            # category_name = self.category_mapping[str(category_id)]
            category_name = "insect"

            # fix negative box coords
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = max(0, bbox[2])
            bbox[3] = max(0, bbox[3])

            # fix out of image box coords
            if full_shape is not None:
                bbox[0] = min(full_shape[1], bbox[0])
                bbox[1] = min(full_shape[0], bbox[1])
                bbox[2] = min(full_shape[1], bbox[2])
                bbox[3] = min(full_shape[0], bbox[3])

            # ignore invalid predictions
            if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                print(f"ignoring invalid prediction with bbox: {bbox}")
                continue

            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=category_id,
                score=score,
                segmentation=None,
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)
        object_prediction_list_per_image.append(object_prediction_list)

    return object_prediction_list_per_image

def get_prediction(
    image,
    detection_model,
    conf_thres,
    iou_thres,
    max_det,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
    exclude_classes_by_name: Optional[List[str]] = None,
    exclude_classes_by_id: Optional[List[int]] = None,
) -> PredictionResult:
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    
    durations_in_seconds = dict()

    # Convert to cv2
    if isinstance(image, np.ndarray):
        im0 = image[:, :, ::-1].copy()
    elif isinstance(image, str):
        im0 = cv2.imread(str(image))
    # get prediction
    time_start = time.time()
    
    # Preprocess
    im = letterbox(im0, args.imgsz, stride=detection_model.stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(detection_model.device)
    im = im.half() if detection_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = detection_model(im, augment=False, visualize=False)
    pred = pred[0][1]

    # NMS
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        # s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # # Write results
            # for *xyxy, conf, cls in reversed(det):
            #     # xyxy coordinates instead of uv coordinates
            #     print(f'xyxy: {xyxy}')
            #     print(f'conf: {conf}')
            #     print(f'cls: {cls}')
    
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if full_shape is None:
        full_shape = [im.height, im.width]
    print(f'full shape: {full_shape}')


    # process prediction
    time_start = time.time()
    # works only with 1 batch
    
    # Convert predictions to the original object
    object_prediction_list: List[ObjectPrediction] = create_object_prediction_list_from_original_predictions(
        pred,
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
    object_prediction_list = filter_predictions(object_prediction_list[0], exclude_classes_by_name, exclude_classes_by_id)

    # postprocess matching predictions
    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )



def main(args):
    # System path definition
    ROOT = 'frameworks/yolov9'
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH

    from models.common import DetectMultiBackend

    # Model loader
    detection_model = DetectMultiBackend(args.weights_path, 
                                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                         dnn=False, data=None, fp16=False)

    # Image path
    durations_in_seconds = dict()
    time_start = time.time()
    
    num_batch = 1
    image = str(args.image_path)
    slice_height = args.slice_window
    slice_width = args.slice_window
    overlap_height_ratio = args.slice_overlap_ratio
    overlap_width_ratio = args.slice_overlap_ratio
    merge_buffer_length = None
    perform_standard_pred = True
    slice_image_result = slice_image(
        image=image,
        output_file_name=None,
        output_dir=None,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=True,
    )
    num_slices = len(slice_image_result)
    print(f'The number of total slices: {num_slices}')
    
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end
    
    ### TODO: argparse 
    # Define post-process methods
    postprocess_type = "NMS"
    postprocess_match_metric = "IOS"
    postprocess_match_threshold = args.slice_match_iou
    postprocess_class_agnostic = False
    
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )
    
    num_group = int(num_slices / num_batch)

    # create prediction input
    object_prediction_list = []
    # perform sliced prediction
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
            exclude_classes_by_name=None,
            exclude_classes_by_id=None,
        )
        # convert sliced predictions to full predictions
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:  # if not empty
                object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            object_prediction_list = postprocess(object_prediction_list)

    # perform standard prediction
    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            shift_amount=[0, 0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
            postprocess=None,
            exclude_classes_by_name=None,
            exclude_classes_by_id=None,
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)

    # merge matching predictions
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)


    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end
    
    if True:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
            )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
    
    result =  PredictionResult(
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )
    
    # export_dir = "outputs/"
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    # Export with custom visualization parameters
    visualize_object_predictions(
        image=np.ascontiguousarray(result.image),
        object_prediction_list=result.object_prediction_list,
        text_size=1.0,  # Size of the class label text
        rect_th=2,      # Thickness of bounding box lines
        text_th=2,      # Thickness of the text
        hide_labels=False,  # Set True to hide class labels
        hide_conf=False,    # Set True to hide confidence scores
        color=(255, 0, 0),  # Custom color in RGB format (red in this example)
        output_dir=args.save_path,
        file_name="custom_visualization",
        export_format="jpg"  # Supports 'jpg' and 'png'
    )


def parser():
    parser = argparse.ArgumentParser("SAHI + YOLOv9 Prediction")
    
    parser.add_argument("--weights-path", type=str, default='./models/best.pt', help="Path to the model weights")
    parser.add_argument("--image-path", type=str, default='./images/input.jpg', help="Path to the image")
    parser.add_argument("--imgsz", '--img', '--img-size', nargs='+', type=int, default=[640], help="Image size")
    parser.add_argument('--conf-thres', type=float, default=0.3, help='Model confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='Model iou threshold')
    
    parser.add_argument('--slice-window', type=int, default=2048, help='Slicing window size')
    parser.add_argument('--slice-overlap-ratio', type=float, default=0.2, help='Slicing window overlap ratio')
    parser.add_argument('--slice-match-iou', type=float, default=0.5, help='Slicing windows match iou threshold')
    
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument("--save-path", type=str, default='outputs/', help="Path to the saved image")
    
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args


if __name__ == "__main__":
    args = parser()
    main(args)
