import os.path as osp
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_prediction


detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/hieu/hieunm/Paddle_parseq/parseq_rec_model/best.pt",
    confidence_threshold=0.1,
    device="cuda:0",
)
# img_paths = glob("/home/hieu/hieunm/POPDetection/Data/shelves/*") + glob("./Data/image/*")
# img_paths = glob("/home/hieu/Downloads/20250203_棚画像_棚読み込み用-20250203T020307Z-001/20250203/*")
img_paths = ["/media/hieu/data/download/北海道写真SavedImages20240402/北海道写真SavedImages/20250318_152432_001origin.png"]
for img_path in img_paths:
    result = get_prediction(img_path,
                            detection_model)
    result.export_visuals("/home/hieu/hieunm/Paddle_parseq/results/result_yolo",
                          hide_conf=True, hide_labels=True,
                          file_name=osp.basename(img_path))