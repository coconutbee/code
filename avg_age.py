import argparse
import logging
import os
import csv
import cv2
import torch
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default='MiVOLO/output', help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default='MiVOLO/model/yolov8x_person_face.pt', help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", type=str, default='MiVOLO/model/model_imdb_cross_person_4.22_99.46.pth.tar', help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    input_type = get_input_type(args.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in args.input:
            args.input, res, fps, yid = get_direct_video_url(args.input)
            if not args.input:
                raise ValueError(f"Failed to get direct video url {args.input}")
            outfilename = os.path.join(args.output, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(args.input))[0]
            outfilename = os.path.join(args.output, f"out_{bname}.avi")
            res, fps = get_local_video_info(args.input)

        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        for (detected_objects_history, frame) in predictor.recognize_video(args.input):
            if args.draw:
                out.write(frame)

    elif input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]

        all_ages = []  

        for img_p in image_files:
            print(f"Process {img_p}")
            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            ages = [a for a in detected_objects.ages if a is not None]
            all_ages.extend(ages)

            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")

        # 处理完所有图片后，计算平均年龄
        if all_ages:
            avg_age = sum(all_ages) / len(all_ages)
        else:
            avg_age = float('nan')
            print("No ages detected in any image.")
        
        folder_name = args.input.split("/")[-1]
        csv_folder = 'MiVOLO/selfage_avg_age'
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder, exist_ok=True)
        csv_path = os.path.join(csv_folder, f"{folder_name}_age.csv")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["average_age"])
            writer.writerow([f"{avg_age:.2f}"])

        print(f"Average age over folder: {avg_age:.2f} → saved to {csv_path}")
        print(f"Average age: {avg_age:.2f}")



if __name__ == "__main__":
    main()
