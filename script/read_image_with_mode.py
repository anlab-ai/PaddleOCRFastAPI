import argparse
import os
import sys
import traceback

# Model paths in ImageReader are relative to the repo root
CALLER_CWD = os.getcwd()
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from models.OCRModel import ImageReader

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def collect_images(input_path):
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(IMAGE_EXTS)
        )
    return [input_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReadImageWithMode on an image or a folder of images")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output", required=True, help="Folder to save result images")
    parser.add_argument("--mode", default=None, help="Read mode (3: filter low-confidence boxes and texts)")
    parser.add_argument("--infos", default=None, help="Infos JSON string (currently unused by ReadImageWithMode)")
    # Also parsed by ImageReader via parse_known_args
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.join(CALLER_CWD, args.input))
    output_dir = os.path.abspath(os.path.join(CALLER_CWD, args.output))
    os.makedirs(output_dir, exist_ok=True)

    image_reader = ImageReader()

    img_paths = collect_images(input_path)
    for img_path in img_paths:
        print(f"Processing: {img_path}")
        with open(img_path, "rb") as f:
            file_bytes = f.read()
        try:
            output_bytes = image_reader.ReadImageWithMode(file_bytes, args.mode, args.infos)
        except Exception:
            print(traceback.format_exc())
            continue

        name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{name}.png")
        with open(save_path, "wb") as f:
            f.write(output_bytes)
        print(f"Saved: {save_path}")
