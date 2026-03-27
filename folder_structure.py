#helper functions for folder structure of the dataset

import os

# folder structure:
    # dataset_name
    #     - videos
    #       - video1.mp4
    #       - video2.mp4
    #       ...
    #     - roi
    #       - video1
    #         - frame_0001.txt
    #         - frame_0002.txt
    #         ...
    #       - video2
    #         - frame_0001.txt
    #         - frame_0002.txt
    #         ...
    #       ...
    #     - frame
    #       - video1
    #         - frame_0001.jpg
    #         - frame_0002.jpg
    #         ...
    #       - video2
    #         - frame_0001.jpg
    #         - frame_0002.jpg
    #         ...
    #       ...
    #     - text_crop
    #       - video1
    #         - frame_0001.jpg
    #         - frame_0002.jpg
    #         ...
    #       - video2
    #         - frame_0001.jpg
    #         - frame_0002.jpg
    #         ...
    #       ...

# get video names without extension from the videos folder
def get_video_name_list(dataset_name, with_ext=False):
    video_folder = os.path.join(dataset_name, "videos")

    def is_valid_video_file(file_name):
        # Ignore hidden files and macOS AppleDouble metadata files (._*)
        if file_name.startswith('.') or file_name.startswith('._'):
            return False
        return file_name.lower().endswith('.mp4')

    if with_ext:
        video_list = [f for f in os.listdir(video_folder) if is_valid_video_file(f)]
    else:
        video_list = [os.path.splitext(f)[0] for f in os.listdir(video_folder) if is_valid_video_file(f)]
    return video_list

def get_video_folder(dataset_name):
    return os.path.join(dataset_name, "videos")

def get_video_path(dataset_name, video_name):
    return os.path.join(get_video_folder(dataset_name), video_name)

def get_roi_folder(dataset_name, video_name):
    return os.path.join(dataset_name, "roi", video_name)

def get_frame_folder(dataset_name, video_name):
    return os.path.join(dataset_name, "frame", video_name)

def get_text_crop_folder(dataset_name, video_name):
    return os.path.join(dataset_name, "text_crop", video_name)

def create_folder_structure(dataset_name):
    # videos and roi should already exist
    #os.makedirs(os.path.join(dataset_name, "videos"), exist_ok=True)
    #os.makedirs(os.path.join(dataset_name, "roi"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, "frame"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, "text_crop"), exist_ok=True)

    # create subfolders for each video in frame and text_crop folders
    video_names = get_video_name_list(dataset_name)
    for video_name in video_names:
        os.makedirs(get_frame_folder(dataset_name, video_name), exist_ok=True)
        os.makedirs(get_text_crop_folder(dataset_name, video_name), exist_ok=True)
