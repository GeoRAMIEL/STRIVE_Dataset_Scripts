import paddleocr
from paddleocr import PaddleOCR
import os
import json
import cv2
import matplotlib.pyplot as plt
from parse import *
import tqdm
import argparse

def video_ocr(video_path, output_file):
    # clear output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist. Please check the path and try again.")
        exit(1)
    
    vidcap = cv2.VideoCapture(video_path)
    # print some info about videocap
    print(f"Video opened: {vidcap.isOpened()}")
    print(f"Video frame count: {vidcap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Video frame rate: {vidcap.get(cv2.CAP_PROP_FPS)}")
    print(f"Video width: {vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Video height: {vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    success, image = vidcap.read()
    count = 0

    if not success:
        print("Error: Could not read the video file. Please check the path and try again.")
        exit(1)

    pbar = tqdm.tqdm()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar.set_description("Processing video frames")
    pbar.reset(total=frame_count)
    while success:
        result = ocr.predict(input=image)
        cur_frame_ocr_results = []
        for res in result:
            rec_texts = res['rec_texts']
            rec_boxes = res['rec_boxes']
            rec_scores = res['rec_scores']
            score_threshold = 0.9
            for text, box, score in zip(rec_texts, rec_boxes, rec_scores):
                if score >= score_threshold:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    cur_frame_ocr_results.append(f"'{text}'  [{x_min},{y_min},{width},{height}]")
        
        with open(output_file, "a") as f:
            f.write(f"[{count}]\n")
            for line in cur_frame_ocr_results:
                f.write(line + "\n")

        success, image = vidcap.read()
        pbar.update(1)

        count += 1
    
    pbar.close()
    print("Frames extracted successfully")

# extracting frames from videos
if __name__ == "__main__":
    # parse input video path and output path
    parser = argparse.ArgumentParser(description="Extract frames from video and perform OCR.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the OCR results.")
    args = parser.parse_args()

    # output file format
    # [0]
    # 'Awoke'  [1450, 90, 328, 146]
    # 'villa'  [774, 283, 187, 98]
    # 'sloop'  [351, 398, 138, 76]
    # 'Vulva'  [1485, 766, 337, 94]
    # 'SHRED'  [769, 785, 187, 64]
    # 'REARM'  [331, 794, 128, 54]
    # [1]
    # 'Awoke'  [1450, 90, 328, 146]
    # 'villa'  [774, 283, 187, 98]
    # 'sloop'  [351, 398, 138, 76]
    # 'Vulva'  [1485, 766, 337, 94]
    # [2]
    # ...

    if args.video_path is None or args.output_file is None:
        print("Error: Please provide both --video_path and --output_file arguments.")
        exit(1)

    video_path = args.video_path
    output_file = args.output_file

    video_ocr(video_path, output_file)

