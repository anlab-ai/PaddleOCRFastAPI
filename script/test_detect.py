from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("./parseq_rec_model/best.pt")

    # img_path = "/media/hieu/data/source_code/RedChalk/2026.10.13-SHIAGEBA_B-01.JPG"
    # img_path = "/media/hieu/data/source_code/RedChalk/debug/201_0.jpg"
    img_path = ""
    result = model(img_path, conf=0.2)
    result[0].save("")