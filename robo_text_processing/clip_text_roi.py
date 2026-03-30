import os
import cv2
import argparse
from parse import *

# Task: Using the output of video_ocr, crop the text regions from the video frames and save them as images.
# There should be one folder for each word that appears in consecutive frames, named {word}_{index}, where index is 
# the ith occurrence of that word in the video. For example, if the word "Hello" appears in frames 10-20, then there 
# should be a folder named "Hello_00" that contains the cropped images of that word from frames 10-20. If the word
# "Hello" appears again in frames 50-60, then there should be another folder named "Hello_01" that contains the cropped
# images of that word from frames 50-60.
# The cropped images should be saved in that folder, named crop_frame_{frame_number}.

# video lengths are around 20-30 minutes, and the frame rate is 30 fps, so there are around 36,000 to 54,000 frames per video.
# we reserve 6 digits for all frame numbers in the folder names and file names

# The output of video_ocr is a text file with the following format:
# [0]
# 'text1'  [x_min,y_min,width,height]
# 'text2'  [x_min,y_min,width,height]
# [1]
# 'text1'  [x_min,y_min,width,height]
# 'text2'  [x_min,y_min,width,height]
# 'text3'  [x_min,y_min,width,height]
# [2]
# 'text2'  [x_min,y_min,width,height]
# 'text3'  [x_min,y_min,width,height]
# ...

# input accepts a WordList csv file, with the first column being the word, and the other columns should be ignored.
# This serves as a whitelist of words to crop, and only those words will be cropped from the video frames.

# define a class for tracking the occurrence of each word in the video, with the following attributes:
class WordTracker:
    def __init__(self, word, occr_reset_threshold=5):
        self.word = word
        self.occr_index = -1
        self.start_frame = -1
        self.last_seen_frame = -occr_reset_threshold - 1
        self.folder_path = None
    
    def inc_occurence(self, new_occr_frame_num):
        self.occr_index += 1
        self.start_frame = new_occr_frame_num
        self.last_seen_frame = new_occr_frame_num

def clip_text_roi(video_path, video_ocr_results_txt, word_list_csv, output_folder, occr_reset_threshold=5, crop_margin=10):
    vidcap = cv2.VideoCapture(video_path)
    # print some info about videocap
    print(f"Video opened: {vidcap.isOpened()}")
    print(f"Video frame count: {vidcap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Video frame rate: {vidcap.get(cv2.CAP_PROP_FPS)}")
    print(f"Video width: {vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Video height: {vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    video_frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # read the word list from the csv file
    word_list = []
    with open(word_list_csv, "r") as f:
        for line in f:
            word = line.split(",")[0].strip()
            word_list.append(word)

    # keep record of the current words that are being tracked, their occurrence indices, and their start frame number
    # words that lost their track for more than 5 frames will be considered as disappeared, and their occurrence will 
    # be ended (occr_index will be updated, and start_frame will be reset to -1)
    # all words are always kept in the current tracking list, and their occurrence indices and start frame numbers are updated accordingly
    # data structure: {word: {"occr_index": int, "start_frame": int, "last_seen_frame": int}}

    # initialize the tracking dictionary using all the words in the word list, with occurrence index 0 and start frame -1
    tracking_dict = {}
    for word in word_list:
        tracking_dict[word] = WordTracker(word, occr_reset_threshold)
    
    cur_frame_num = 0
    cur_video_image = None
    with open(video_ocr_results_txt, "r") as f:
        cur_line = f.readline()
        while cur_line:
            if cur_line.strip():
                cur_line = cur_line.strip()

            if cur_line.startswith("["):
                # this is a frame number line
                ss = parse("[{}]", cur_line)
                if not ss:
                    print(f"Warning: Failed to parse frame number line: {cur_line}. Skipping.")
                    cur_line = f.readline()
                    continue
                cur_frame_num = int(ss[0])
                cur_line = f.readline()
                # read the next frame of the video
                frame_num_in_video = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_num_in_video != cur_frame_num:
                    print(f"Warning: Frame number in OCR results ({cur_frame_num}) does not match the current frame number in video ({frame_num_in_video}). Seeking to the correct frame.")
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_num)
                success, cur_video_image = vidcap.read()
                if not success:
                    print("Error: Could not read the video file. Please check the path and try again.")
                    exit(1)
                continue

            # parse the next line as an OCR result line
            ss = parse("'{}'  [{},{},{},{}]", cur_line)
            if not ss:
                cur_line = f.readline()
                continue
            text = ss[0]
            x_min = int(ss[1])
            y_min = int(ss[2])
            width = int(ss[3])
            height = int(ss[4])

            # add margin to the cropped region and crop inside the video frame boundaries
            x_min = max(0, x_min - crop_margin)
            y_min = max(0, y_min - crop_margin)
            x_max = min(video_frame_width, x_min + width + 2 * crop_margin)
            y_max = min(video_frame_height, y_min + height + 2 * crop_margin)

            if text in tracking_dict:
                # this is a word we are tracking

                # compare cur_frame_num with last_seen_frame to determine if the word is still being tracked or
                # if it has disappeared and reappeared
                if cur_frame_num - tracking_dict[text].last_seen_frame > occr_reset_threshold:
                    # the word has disappeared for more than 5 frames, consider it as disappeared and update the 
                    # occurrence index and reset the start frame
                    # this line should also be executed for the first occurrence of the word
                    tracking_dict[text].inc_occurence(cur_frame_num)
                    # create a new folder for the new occurrence of the word, reserve 2 digits for the occurrence index
                    new_folder_name = f"{text}_{tracking_dict[text].occr_index:02d}"
                    new_folder_path = os.path.join(output_folder, new_folder_name)
                    os.makedirs(new_folder_path, exist_ok=True)
                    tracking_dict[text].folder_path = new_folder_path

                # this is a word that is currently being tracked, update the last seen frame
                tracking_dict[text].last_seen_frame = cur_frame_num
                # crop the text region from the video frame and save it as an image in the corresponding folder
                if cur_video_image is not None:
                    text_region = cur_video_image[y_min:y_max, x_min:x_max]
                    new_image_path = os.path.join(tracking_dict[text].folder_path, f"crop_frame_{cur_frame_num:06d}.jpg")
                    cv2.imwrite(new_image_path, text_region)
            else:
                # this is a word we are not tracking, ignore it
                pass
            cur_line = f.readline()
    
    # release the video capture object
    vidcap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip text regions from video frames based on OCR results.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--ocr_results", type=str, required=True, help="Path to the OCR results text file.")
    parser.add_argument("--word_list_csv", type=str, required=True, help="Path to the word list CSV file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the cropped text region images.")
    parser.add_argument("--occr_reset_threshold", type=int, default=5, help="Number of frames to consider a word as disappeared if not seen.")
    parser.add_argument("--crop_margin", type=int, default=10, help="Margin to add around the cropped text region.")
    args = parser.parse_args()

    clip_text_roi(args.video_path, args.ocr_results, args.word_list_csv, args.output_folder, args.occr_reset_threshold, args.crop_margin)
    print("Text region cropping completed successfully.")
