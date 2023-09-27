# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 10:52
# @Author  : yblir
# @File    : split_videos_scenes.py
# explain  : 
# =======================================================
# -*- coding: utf-8 -*-
import os
import sys
# import argparse
from pathlib2 import Path
from loguru import logger

# import cv2
# import numpy as np
import torch
# import math
import warnings
# import albumentations as alb
# from albumentations.pytorch.transforms import ToTensorV2
# import time
# import yaml
import shutil

# from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import SceneManager, VideoStreamCv2

from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

# from models import *
# from common.utils import *
# from scrfd_opencv_gpu.scrfd_face_detect import SCRFD, get_max_face_box
# from decord import VideoReader
# import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)

if __name__ == '__main__':
    # 原视频路径
    root_path = Path(r"E:\DeepFakeDetection\mabaog")
    # 分割后保存路径
    split_save_dir = Path(r"E:\DeepFakeDetection\mabaog_scenes")

    os.makedirs(str(split_save_dir), exist_ok=True)

    for video_path in root_path.iterdir():
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video_manager = VideoStreamCv2(str(video_path))
        scene_manager.detect_scenes(frame_source=video_manager)

        # time_dic = {}
        # 保存为视频文件
        scene_list = scene_manager.get_scene_list()
        if not scene_list:
            shutil.copy2(str(video_path), str(split_save_dir))
            continue
        for index, scene in enumerate(scene_list):
            # split_start_time = scene[0].get_timecode().replace("00:", "", 1)[:8]
            # split_end_time = scene[1].get_timecode().replace("00:", "", 1)[:8]
            split_video_ffmpeg(str(video_path), [scene],
                               # rf"C:\Users\Administrator\Desktop\youtube\xidada_\{index + 1}.mp4",
                               f"{split_save_dir}/{str(video_path.stem)}_{index}.mp4",
                               "",
                               # show_progress=True,
                               # show_output=True,
                               # suppress_output=True
                               )
            logger.info(f"scene over: {index}")
            # t=split_end_time.split("00:")[0]
            # scene_time = float(split_end_time.split("00:")[1]) - float(split_start_time.split("00:")[1])
            # time_dic[index] = (split_start_time, split_end_time, round(scene_time, 3))

    logger.success("success")
