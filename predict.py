# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 9:16
# @Author  : yblir
# @File    : predict.py
# explain  : 
# =======================================================
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib2 import Path
from loguru import logger

import cv2
import numpy as np
import torch
import math
import warnings
# import albumentations as alb
# from albumentations.pytorch.transforms import ToTensorV2
import time

from models import *
# from common.utils import *
from scrfd_opencv_gpu.scrfd_face_detect import SCRFD, get_max_face_box
from decord import VideoReader
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)

num_segments = 16
margin = 1.3
# h_margin = 1.1
# w_margin = 1.2
sparse_span = 150

# with open("configs/ffpp.yaml", "r", encoding="utf-8") as f:
#     args = yaml.safe_load(f)
# args = get_params()

# # set model and wrap it with DistributedDataParallel
# model = eval(args.model.name)(**args.model.params)
# model.set_segment(args.test.dataset.params.num_segments)
# model.cuda()
# model.eval()
###############


# transform = alb.Compose([
#     alb.Resize(224, 224),
#     alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ], additional_targets={})

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


# ==============================================================================

def get_enclosing_box(img_h, img_w, box, margin):
    """Get the square-shape face bounding box after enlarging by a certain margin.

    Args:
        img_h (int): Image height.
        img_w (int): Image width.
        box (list): [x0, y0, x1, y1] format face bounding box.
        margin (float): The margin to enlarge.

    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    max_size = max(w, h)

    cx = x0 + w / 2
    cy = y0 + h / 2
    x0 = cx - max_size / 2
    y0 = cy - max_size / 2
    x1 = cx + max_size / 2
    y1 = cy + max_size / 2

    offset = max_size * (margin - 1) / 2
    x0 = int(max(x0 - offset, 0))
    y0 = int(max(y0 - offset, 0))
    x1 = int(min(x1 + offset, img_w))
    y1 = int(min(y1 + offset, img_h))

    return [x0, y0, x1, y1]


def face_allign(landmarks, img):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.atan2(dy, dx) * 180. / math.pi
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))

    return rotated_img


def sample_indices_test(vr):
    """Frame sampling strategy in test stage.

    Args:
        video_len (int): Video frame count.

    """
    video_len = len(vr)

    base_idxs = np.linspace(0, video_len - 1, sparse_span, dtype=np.int)
    base_idxs_len = len(base_idxs)

    tick = base_idxs_len / float(num_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    offsets = base_idxs[offsets].tolist()

    return base_idxs, offsets


def get_faces_from_selected_frames(vr, base_idxs, sampled_idxs, mynet):
    img_h, img_w, _ = vr[0].shape
    frames = vr.get_batch(sampled_idxs).asnumpy()
    imgs = []
    for idx in range(len(frames)):
        img = frames[idx]
        try:
            res = mynet.detect(img)
            if not res:
                raise
        except Exception as _:
            raise ValueError("face detect failure")

        output_box, kpss = get_max_face_box(res)
        x0, y0, x1, y1 = output_box

        x0, y0, x1, y1 = get_enclosing_box(img_h, img_w, [x0, y0, x1, y1], margin)
        img = img[y0:y1, x0:x1]

        imgs.append(img)

    return imgs


def model_infer(model, video_path, mynet):
    # frames = video_preprocess(str(video_path), mynet)

    with torch.no_grad():
        # images = frames.cuda(args.local_rank).unsqueeze(0)
        images = frames.cuda().unsqueeze(0)
        outputs = model(images)
    real_probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()[0]

    predict_dict = {"face": "fake" if real_probs <= 0.5 else "real",
                    "prob": (1 - real_probs).tolist() if real_probs <= 0.5 else real_probs.tolist()}
    return predict_dict


def get_video_label(video_path):
    temp_path = str(video_path)
    # logger.info(f'======={video_id}======')
    if 'fake' in temp_path:
        label = torch.tensor([1]).cuda(0)
    else:
        label = torch.tensor([0]).cuda(0)
    return label


def get_file_nums(path):
    mp4_list = []
    for roots, dirs, files in os.walk(path):
        for mp4_file in files:
            mp4_list.append(mp4_file)
    file_num = len(mp4_list)
    return file_num


def draw_fake_video(draw_path, mynet, face_flag):
    """
    对fake片段绘图
    """
    cap = cv2.VideoCapture(str(draw_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
            str(draw_path.parent / (str(draw_path.stem) + "_draw.mp4")), fourcc, fps, (img_w, img_h)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = mynet.detect(frame)
        if res:
            output_box, kpss = get_max_face_box(res)
            x0, y0, x1, y1 = [int(i) for i in output_box]
            cv2.rectangle(frame,
                          (x0, y0), (x1, y1),
                          (0, 0, 255) if face_flag == "fake" else (0, 255, 0),
                          3)
        video_writer.write(frame)

    cap.release()
    video_writer.release()


def run_thread(id, split_video_paths, res_dic, LOCK):
    print(f"start process  id: {id}")
    # record_fake = {}
    infer_info = {}
    model = STILModel()

    model.set_segment(16)

    model.cuda()
    model.eval()

    mynet = SCRFD('scrfd_opencv_gpu/weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)
    ckpt_load_path = r"E:\DeepFakeDetection\STIL\models\内网X8149TO外网X8149_1688369358623.tar"
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')

    new_checkpoint = {}

    for k, value in checkpoint["state_dict"].items():
        new_checkpoint[k.split("module.")[-1]] = value

    model.load_state_dict(new_checkpoint)

    for split_video_path in split_video_paths:
        try:
            predict_dic = model_infer(model, split_video_path, mynet)
        except ValueError as e1:
            # record_fake[f"{split_video_path.stem}"] = 0
            infer_info[split_video_path.stem] = ["None", str(e1)]
            continue
        except Exception as e2:
            logger.error(f"{str(split_video_path.name)} error:{e2}")
            sys.exit()

        # 画出换脸片段real或fake区域
        # record_fake[f"{split_video_path.stem}"] = 1 if predict_dic["face"] == "fake" else 0
        draw_fake_video(split_video_path, mynet, predict_dic["face"])

        # 合并推理信息
        infer_info[split_video_path.stem] = [
            "fake" if predict_dic["face"] == "fake" else "real", round(predict_dic["prob"], 3)
        ]

    with LOCK:
        for key in infer_info.keys():
            if key in res_dic.keys():
                item = res_dic[key]
                item.extend(infer_info[key])
                res_dic[key] = item


def stil_predict(path2):
    t1 = time.time()
    video_path = Path(path2)
    os.makedirs("./temp_video", exist_ok=True)

    if video_path.suffix.lower().strip() != ".mp4":
        logger.error(f"video suffix is not mp4, and is {video_path.suffix}")
        sys.exit()

    # 获得每个时间片段时间信息
    # infer_info = find_scenes_save_video(video_path)

    if not infer_info:
        # 若infer_info为空,说明视频没有分段,处理原始视频
        infer_info = {video_path.stem: ["00:00:00", "entire_video"]}
        split_video_list = [video_path]
    else:
        split_video_list = [i for i in Path("./temp_video").iterdir()]
        split_video_list.sort(key=lambda x: str.zfill(x, 5), reverse=False)

    video_segments_nums = len(infer_info)
    # 记录fake片段编号,用于合成新视频时区分
    # record_fake = {}

    # ######################################## multiprocess #####################################
    nprocess_this_node = 2 if len(split_video_list) > 2 else len(split_video_list)
    pool = mp.Pool(nprocess_this_node)

    manager = mp.Manager()
    res_dic = manager.dict()
    res_dic.update(infer_info)
    LOCK = manager.Lock()

    for i in range(nprocess_this_node):
        pool.apply_async(run_thread, (i, split_video_list[i::nprocess_this_node], res_dic, LOCK))
    pool.close()
    pool.join()

    # 合成新视频
    video_list = [None] * video_segments_nums

    for i, split_video_path in enumerate(split_video_list):
        try:
            video_list[i] = VideoFileClip(str(split_video_path.parent / (split_video_path.stem + "_draw.mp4")))
        except:
            video_list[i] = VideoFileClip(str(split_video_path))

    final_video = concatenate_videoclips(video_list)
    save_fake_path = "./temp_video/" + video_path.stem + "_fake.mp4"
    final_video.write_videofile(save_fake_path)

    cost_time = time.time() - t1

    return save_fake_path, res_dic, round(cost_time, 3)


# 4.7512 -4.7717
if __name__ == "__main__":
    model = STILModel()

    model.set_segment(16)

    model.cuda()
    model.eval()

    # ckpt_load_path = r"E:\DeepFakeDetection\STIL\models\85_epoch_model.tar"
    ckpt_load_path = "/mnt/e/DeepFakeDetection/STIL/models/85_epoch_model.tar"
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')

    new_checkpoint = {}

    for k, value in checkpoint["state_dict"].items():
        new_checkpoint[k.split("module.")[-1]] = value

    model.load_state_dict(new_checkpoint)
    # model=torch.jit.script(model)
    # exmple = torch.ones(1, 48, 224, 224).cuda()
    # exmple = []
    # frames2 = []
    # for i in range(16):
    #     img = cv2.imread("/mnt/e/GitHub/pythonProject/test3/tensorrt/aab.jpg")
    #     img = cv2.resize(img, (224, 224))
    #     exmple.append(img)
    # exmple = np.array(exmple)
    exmple = torch.ones(1, 48, 224, 224).cuda()
    torch.onnx.export(
            model,
            # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
            (exmple,),

            # 储存的文件路径
            "stil_linux.onnx",

            # 打印详细信息
            verbose=True,

            # 为输入和输出节点指定名称，方便后面查看或者操作
            input_names=["images"],
            output_names=["output"],

            # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
            opset_version=11,

            # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
            # 通常，我们只设置batch为动态，其他的避免动态
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"},
            }
    )

    print("ok")
