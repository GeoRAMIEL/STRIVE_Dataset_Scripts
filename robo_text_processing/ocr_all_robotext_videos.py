import os
import cv2
from video_ocr import video_ocr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from video frames using OCR.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing video files.")
    args = parser.parse_args()

    video_folder = args.video_folder

    output_folder = "output"

    # get all mp4 files in the video folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_ocr.txt")
        print(f"===== Processing video: {video_path} ---------> Output file: {output_file} =====")
        video_ocr(video_path, output_file)
    
    print("===== All done =====")
