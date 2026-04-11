import os
import argparse
from clip_text_roi import clip_text_roi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip text regions from video frames based on OCR results.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--ocr_result_folder", type=str, required=True, help="Path to the OCR results text file.")
    parser.add_argument("--word_list_csv_folder", type=str, required=True, help="Path to the word list CSV file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the cropped text region images.")
    parser.add_argument("--occr_reset_threshold", type=int, default=5, help="Number of frames to consider a word as disappeared if not seen.")
    parser.add_argument("--crop_margin", type=int, default=10, help="Margin to add around the cropped text region.")
    args = parser.parse_args()

    video_folder = args.video_folder
    ocr_result_folder = args.ocr_result_folder
    word_list_csv_folder = args.word_list_csv_folder
    output_folder = args.output_folder
    occr_reset_threshold = args.occr_reset_threshold
    crop_margin = args.crop_margin

    # get all mp4 files in the video folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_ocr_results_txt = os.path.join(ocr_result_folder, f"{os.path.splitext(video_file)[0]}_ocr.txt")
        word_list_csv = os.path.join(word_list_csv_folder, f"{os.path.splitext(video_file)[0]}_WordList.csv")
        # verify the paths exist
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist. Please check the path and try again.")
            continue
        if not os.path.exists(video_ocr_results_txt):
            print(f"Error: The OCR results file '{video_ocr_results_txt}' does not exist. Please check the path and try again.")
            continue
        if not os.path.exists(word_list_csv):
            print(f"Error: The word list CSV file '{word_list_csv}' does not exist. Please check the path and try again.")
            continue
        # create output folder for the video
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
        os.makedirs(video_output_folder, exist_ok=True)
        print(f"===== Processing video: {video_path} --> Output folder: {video_output_folder} =====")
        clip_text_roi(video_path, video_ocr_results_txt, word_list_csv, video_output_folder, occr_reset_threshold, crop_margin)