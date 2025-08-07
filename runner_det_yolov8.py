from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from sahi.utils.cv import visualize_object_predictions
from pathlib import Path
import numpy as np

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path="/home/yu/Documents/repos/flat-bug/examples/tutorials/flat_bug_M.pt") # for YOLOv8/YOLO11/YOLO12 models

# get sliced prediction result
result = get_sliced_prediction(
    "/home/yu/Documents/AMMOD/filtered-nid-images/2020_08_08_Lichtfang_Hahnengrund_6875.JPG",
    detection_model,
    slice_height = 2048,
    slice_width = 2048,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)


Path("outputs").mkdir(parents=True, exist_ok=True)
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
    output_dir="outputs",
    file_name="custom_visualization_yolov8",
    export_format="jpg"  # Supports 'jpg' and 'png'
)
