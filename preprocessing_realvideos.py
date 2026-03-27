import cv2
import os
import matplotlib.pyplot as plt
from parse import *
import folder_structure as fs
import tqdm

# extracting frames from videos

# code modified to follow the actual folder structure of the dataset (as in run_all.py)

def extract_frame(folderPath, videoName):
    video_stem = os.path.splitext(videoName)[0]
    target_dir = fs.get_frame_folder(folderPath, video_stem)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    video_path = fs.get_video_path(folderPath, videoName)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    pbar = tqdm.tqdm()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar.reset(total=frame_count)
    while success:
        cv2.imwrite(target_dir + "/frame_%04d.jpg" % count,
                    image)  # save frame as JPEG file
        success, image = vidcap.read()
        pbar.update(1)
        # print(count, ': Read a new frame: ', success)

        count += 1

    return "Frames extracted successfully"


# Parsing the bbox file and get the text crop from the videos

def get_bb(filename):
    x = 0
    y = 0
    hei = 0
    wid = 0
    # print(filename.replace('.jpg','.txt'))


    if not os.path.exists(filename): return x, y, hei, wid
    count = len(open(filename).readlines())
    fp = open(filename, "r")
    for i in range(count):
        line = fp.readline()
    ss = parse("'{}'  [{},{},{},{}]", line)
    # ss = parse("'{}' [{} {} {} {}]", line)

    if ss == None: return 0, 0, 0, 0
    x = int(float(ss[1]))  # size_img[0] - int(float(ss[2]))
    y = int(float(ss[2]))
    wid = int(float(ss[3])) - int(float(ss[1]))
    hei = int(float(ss[4])) - int(float(ss[2]))
    # x = x- hei
    return x, y, wid, hei

# retrieving and storing text crops

def extract_bg_crop(folderPath):
    video_name_list = fs.get_video_name_list(folderPath)

    for video_name in video_name_list:
        roi_ann_folder = fs.get_roi_folder(folderPath, video_name)
        frame_folder = fs.get_frame_folder(folderPath, video_name)
        text_crop_folder = fs.get_text_crop_folder(folderPath, video_name)

        if not os.path.exists(text_crop_folder):
            os.mkdir(text_crop_folder)

        def is_valid_img_file(file_name):
            # Ignore hidden files and macOS AppleDouble metadata files (._*)
            if file_name.startswith('.') or file_name.startswith('._'):
                return False
            return file_name.lower().endswith('.jpg')
        images = [i for i in os.listdir(frame_folder) if is_valid_img_file(i)]

        pbar = tqdm.tqdm(images)
        for image in pbar:
            #print(image)
            pbar.set_description(f"Processing {video_name} - {image}")
            txt_file = os.path.join(roi_ann_folder, image.replace('.jpg', '.txt'))
            # txt_file may not exist
            if not os.path.exists(txt_file):
                continue
            x, y, wid, hei = get_bb(txt_file)
            if wid == 0 or hei == 0:
                continue
            img_name = os.path.join(frame_folder, image)
            I = cv2.imread(img_name, cv2.COLOR_BGR2RGB)
            crop = I[y:y + hei, x:x + wid]

            ## store the crop for further analysis
            cv2.imwrite(os.path.join(text_crop_folder, "crop_" + image), crop)
