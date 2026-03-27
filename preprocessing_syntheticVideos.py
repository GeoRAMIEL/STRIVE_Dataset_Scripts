import cv2
import os
import matplotlib.pyplot as plt
from parse import *
import folder_structure as fs
import tqdm

# the original annatations are a mess, need to define different parsing rules for different video series

# case 0: default case, text format: 'spot' [869.1163 328.0847 164.1194 355.8832], numbers are: xmin, ymin, width, height

# case 1: Highway series, text format: 'chicago'  [927,403,990,423], numbers are xmin, ymin, xmax, ymax

# case 2: SchoolCorridorN-M series, text format: 'Glands' [633.0847,515.972,44.90509,43.22626], numbers are: xmin, ymin, width, height

# extracting frames from videos

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
        cv2.imwrite(os.path.join(target_dir, "IMG_frame_%04d.jpg" % (count + 1)),
                    image)  # save frame as JPEG file
        success, image = vidcap.read()
        pbar.update(1)
        # print(count, ': Read a new frame: ', success)

        count += 1

    return "Frames extracted successfully"

# Get bbox for synthetic videos
def get_bb_synthetic_default_case(filename, img_width=None, img_height=None):
    x = 0
    y = 0
    hei = 0
    wid = 0
    #print('file name : ', filename)

    if not os.path.exists(filename): return x, y, hei, wid
    count = len(open(filename).readlines())
    fp = open(filename, "r")

    last_line = ""
    for i in range(count):
        line = fp.readline()
        #print("read line: ", line)
        if line.strip():  # Check if the line is not empty
            last_line = line.strip()  # Update last_line to the most recent non-empty line

    #print(line)
    #ss = parse("'{}'  [{},{},{},{}]", last_line)
    ss = parse("'{}' [{} {} {} {}]", last_line)
    #print(ss)
    if ss is None:
        print(f"Warning: Failed to parse line in {filename}. Skipping.")
        return 0, 0, 0, 0
    x = int(float(ss[1]))  # size_img[0] - int(float(ss[2]))
    y = img_height - int(float(ss[2])) - int(float(ss[4])) # y coord is flipped
    wid = int(float(ss[3]))
    hei = int(float(ss[4]))
    return x, y, wid, hei

# Get bbox for synthetic videos
def get_bb_synthetic_case1(filename, img_width=None, img_height=None):
    x = 0
    y = 0
    hei = 0
    wid = 0
    #print('file name : ', filename)

    if not os.path.exists(filename): return x, y, hei, wid
    count = len(open(filename).readlines())
    fp = open(filename, "r")

    last_line = ""
    for i in range(count):
        line = fp.readline()
        #print("read line: ", line)
        if line.strip():  # Check if the line is not empty
            last_line = line.strip()  # Update last_line to the most recent non-empty line

    #print(line)
    ss = parse("'{}'  [{},{},{},{}]", last_line)
    #ss = parse("'{}' [{} {} {} {}]", last_line)
    #print(ss)
    if ss is None:
        print(f"Warning: Failed to parse line in {filename}. Skipping.")
        return 0, 0, 0, 0
    x = int(float(ss[1]))  # size_img[0] - int(float(ss[2]))
    y = int(float(ss[2]))
    wid = int(float(ss[3])) - int(float(ss[1]))
    hei = int(float(ss[4])) - int(float(ss[2]))
    return x, y, wid, hei

def get_bb_synthetic_case2(filename, img_width=None, img_height=None):
    x = 0
    y = 0
    hei = 0
    wid = 0
    #print('file name : ', filename)

    if not os.path.exists(filename): return x, y, hei, wid
    count = len(open(filename).readlines())
    fp = open(filename, "r")

    last_line = ""
    for i in range(count):
        line = fp.readline()
        #print("read line: ", line)
        if line.strip():  # Check if the line is not empty
            last_line = line.strip()  # Update last_line to the most recent non-empty line

    #print(line)
    ss = parse("'{}' [{},{},{},{}]", last_line)
    #print(ss)
    if ss is None:
        print(f"Warning: Failed to parse line in {filename}. Skipping.")
        return 0, 0, 0, 0
    x = int(float(ss[1]))  # size_img[0] - int(float(ss[2]))
    y = img_height - int(float(ss[2])) - int(float(ss[4])) # y coord is flipped
    wid = int(float(ss[3]))
    hei = int(float(ss[4]))
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
            pbar.set_description(f"Processing {video_name} - {image}")
            txt_file = os.path.join(roi_ann_folder, image.replace('.jpg', '.txt'))
            if not os.path.exists(txt_file):
                #print(f"Warning: Annotation file {txt_file} does not exist. Skipping.")
                continue

            I = cv2.imread(os.path.join(frame_folder, image))
            img_height, img_width = I.shape[:2]

            if "highway" in video_name.lower():
                x, y, wid, hei = get_bb_synthetic_case1(txt_file, img_width, img_height)
                #print(f"txt_file {txt_file} x: {x}, y: {y}, width: {wid}, height: {hei}")
            elif ("SchoolCorridor" in video_name and not "SchoolCorridor-" in video_name) or "TownhouseProject" in video_name or "TruckScene" in video_name or "WindridgeBoard" in video_name:
                x, y, wid, hei = get_bb_synthetic_case2(txt_file, img_width, img_height)
                #print(f"txt_file {txt_file} x: {x}, y: {y}, width: {wid}, height: {hei}")
            else:
                x, y, wid, hei = get_bb_synthetic_default_case(txt_file, img_width, img_height)
            if hei == 0 or wid == 0:
                #print(f"Warning: Invalid bounding box for image {image}. Skipping.")
                continue

            # original code was retarded
            #top = abs(y)
            #left = abs(x)
            #H = left + (hei)
            #right = left + H
            #W = top + (wid)
            #bottom = top + W
            #padding = 0

            #print(x, y, hei, wid)

            # no idea why they wanted to rotate the video frame because the result is wrong
            #img_rotate = cv2.rotate(I, cv2.ROTATE_90_CLOCKWISE)
            #crop = img_rotate[top:bottom, left:right]

            # crop the box to be inside image boundaries
            x = min(max(0, x), img_width)
            y = min(max(0, y), img_height)
            wid = min(wid, img_width - x)
            hei = min(hei, img_height - y)
            if wid <= 0 or hei <= 0:
                #print(f"Warning: Adjusted bounding box has non-positive width or height for image {image}. Skipping.")
                continue

            crop = I[y:y+hei, x:x+wid]

            # Store the main text crop in the standard output folder structure.
            cv2.imwrite(os.path.join(text_crop_folder, "crop_" + image), crop)


# backward-compatible alias for older entry points
def get_crop_from_videos(folderPath):
    return extract_bg_crop(folderPath)
