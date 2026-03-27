import preprocessing_realvideos
import preprocessing_robotext
import preprocessing_syntheticVideos
import folder_structure as fs

if __name__ == "__main__":
    # extracting frames and text crops from videos

    #print("Starting to process RealWorld videos...")
    #real_world_folder = "/Volumes/Samsung 980/STRIVE_Datasets/RealWorld"
    #fs.create_folder_structure(real_world_folder)
    #video_names_1 = fs.get_video_name_list(real_world_folder)
    #total_video_count_1 = len(video_names_1)
    #for i, video_name in enumerate(video_names_1):
    #    print("Extracting frames from video: ", video_name, " (", i + 1, "/", total_video_count_1, ")")
    #    preprocessing_realvideos.extract_frame(real_world_folder, video_name + '.mp4')
    #preprocessing_realvideos.extract_bg_crop(real_world_folder)

    #print("Starting to process robotext videos...")
    #robotext_folder = "/Volumes/Samsung 980/STRIVE_Datasets/Robotext-1"
    #fs.create_folder_structure(robotext_folder)
    #video_names_2 = fs.get_video_name_list(robotext_folder)
    #total_video_count_2 = len(video_names_2)
    #for i, video_name in enumerate(video_names_2):
    #    print("Extracting frames from video: ", video_name, " (", i + 1, "/", total_video_count_2, ")")
    #    preprocessing_robotext.extract_frame(robotext_folder, video_name + '.mp4')
    #preprocessing_robotext.extract_bg_crop(robotext_folder)

    print("Starting to process SynthText videos...")
    synthetic_folder = "/Volumes/Samsung 980/STRIVE_Datasets/SynthText"
    #fs.create_folder_structure(synthetic_folder)
    video_names_3 = fs.get_video_name_list(synthetic_folder)
    #total_video_count_3 = len(video_names_3)
    #for i, video_name in enumerate(video_names_3):
    #    print("Extracting frames from video: ", video_name, " (", i + 1, "/", total_video_count_3, ")")
    #    preprocessing_syntheticVideos.extract_frame(synthetic_folder, video_name + '.mp4')
    preprocessing_syntheticVideos.extract_bg_crop(synthetic_folder)

    