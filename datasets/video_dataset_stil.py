import os
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
import pickle
import random
from collections import OrderedDict
from pathlib2 import Path
from decord import VideoReader, cpu


class FFPP_Dataset(data.Dataset):
    def __init__(self,
                 root,
                 face_info_path,
                 method='Deepfakes',
                 compression='c23',
                 split='train',
                 num_segments=16,
                 transform=None,
                 sparse_span=150,
                 dense_sample=0,
                 test_margin=1.3):
        """Dataset class for ffpp dataset.

        Args:
            root (str): 
                The root path for ffpp data.
            face_info_path (str): 
                The pickle path containing ffpp face rect info.
            method (str, optional): 
                Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']. 
                Defaults to 'Deepfakes'.
            compression (str, optional): 
                Video compression rate. One of ['c23', 'c40']. Defaults to 'c23'.
            split (str, optional): 
                Data split. One if ['train', 'val', 'test']. Defaults to 'train'.
            num_segments (int, optional): 
                How many frames to choose from each video. Defaults to 16.
            transform (function, optional): 
                Data augmentation. Defaults to None.
            sparse_span (int, optional): 
                How many frames to sparsely select from the whole video. Defaults to 150.
            dense_sample (int, optional): 
                How many frames to densely select. Defaults to 0.
            test_margin (float, optional): 
                The margin to enlarge the face bounding box at test stage. Defaults to 1.3.
        """
        super().__init__()

        self.root = root
        self.face_info_path = face_info_path
        self.method = method
        self.compression = compression
        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.sparse_span = sparse_span
        self.dense_sample = dense_sample
        self.test_margin = test_margin

        assert self.compression in ['c23', 'c40']
        assert self.method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        self.parse_dataset_info()

    def parse_dataset_info(self):
        """Parse the video dataset information
        """
        self.real_video_dir = os.path.join(self.root, 'original_sequences', 'youtube', self.compression, 'videos')
        self.fake_video_dir = os.path.join(self.root, 'manipulated_sequences', self.method, self.compression, 'videos')
        self.split_json_path = os.path.join(self.root, 'splits', f'{self.split}.json')

        assert os.path.exists(self.real_video_dir)
        assert os.path.exists(self.fake_video_dir)
        assert os.path.exists(self.split_json_path)

        # ========================================================================
        # ff++
        json_data = json.load(open(self.split_json_path, 'r'))

        self.real_names = []
        self.fake_names = []
        for item in json_data:
            # 071_054, 071是原视频,把054中人脸截出来替换071中人脸
            self.real_names.extend([item[0], item[1]])
            self.fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])

        # self.dataset_info = [[x, 'real'] for x in self.real_names] + [[x, 'fake'] for x in self.fake_names]
        self.dataset_info = [[x, 'real'] for x in self.real_names]

        for fake_style in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            for x in self.fake_names:
                self.dataset_info.append([x, fake_style])
        # # ==============================================================================
        # # df-1.0 与ff++一样, 测试抽取文件名也与ff+相同
        # df1_real_names, df1_fake_names = [], []
        # df1_root_path = Path("")
        # for dir in ["end_to_end", "reenact_postprocess"]:
        #     for video_path in (df1_root_path / dir).rglob("*"):
        #         if video_path.suffix != ".mp4":
        #             continue
        #         df1_fake_names.append(str(video_path.name))
        # df1_info = [[x, "df1_fake"] for x in df1_fake_names]
        # self.dataset_info.extend(df1_info)
        # # ========================================================================
        # # dfdc
        # dfdc_real_names, dfdc_fake_names = [], []
        # dfdc_json_data = json.load(open("", 'r'))
        # for k, value in dfdc_json_data.items():
        #     if value["label"] == "REAL":
        #         dfdc_real_names.append(k)
        #     else:
        #         dfdc_fake_names.append(k)
        # dfdc_info = [[x, "dfdc_real"] for x in dfdc_real_names] + [[x, "dfdc_fake"] for x in dfdc_fake_names]
        # self.dataset_info.extend(dfdc_info)
        # # ========================================================================
        # # celeb-df,训练时,两个real文件夹混合在一起
        # celeb_real_names, celeb_fake_names = [], []
        # celeb_root_path = Path("")
        # for dir in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
        #     for video_path in (celeb_root_path / dir).rglob("*"):
        #         if video_path.suffix != ".mp4":
        #             continue
        #         if dir in ["Celeb-real", "YouTube-real"]:
        #             celeb_real_names.append(str(video_path.name))
        #         else:
        #             celeb_fake_names.append(str(video_path.name))
        # celeb_info = [[x, "celeb_real"] for x in celeb_real_names] + [[x, "celeb_fake"] for x in celeb_fake_names]
        # self.dataset_info.extend(celeb_info)

        # ========================================================================
        # forgeryNet

        # ========================================================================

        # load face bounding box information.
        self.face_info = pickle.load(open(self.face_info_path, 'rb'))

        print(
                f'{self.split} has {len(self.real_names)} real videos and {len(self.fake_names)} fake videos, face_info has {len(self.dataset_info)}')

    def sample_indices_train(self, video_len, data_list):
        """Frame sampling strategy in training stage.

        Args:
            video_len (int): Video frame length.

        """
        base_idxs = np.array(range(video_len), np.int)
        count = 0
        for i, item in enumerate(data_list):
            if not item or len(item) == 0:
                # print("=====================")
                base_idxs = np.delete(base_idxs, count)
                continue
            count += 1
        real_idxs = base_idxs

        if self.sparse_span:
            base_idxs_index = np.linspace(0, len(base_idxs) - 1, self.sparse_span, dtype=np.int)
            base_idxs = base_idxs[base_idxs_index]
        base_idxs_len = len(base_idxs)

        def over_sample_strategy(total_len):
            if total_len >= self.num_segments:
                offsets = np.sort(random.sample(range(total_len), self.num_segments))
            else:
                inv_ratio = self.num_segments // total_len
                offsets = []
                for idx in range(total_len):
                    offsets.extend([idx] * inv_ratio)
                tail = [total_len - 1] * (self.num_segments - len(offsets))
                offsets.extend(tail)
                offsets = np.asarray(offsets)
            return offsets

        def dense_sample(total_len):
            # print(f'dense! total_len: {total_len}')
            if total_len > self.dense_sample:
                start_idx = np.random.randint(0, total_len - self.dense_sample)
                average_duration = self.dense_sample // self.num_segments
                assert average_duration > 1
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=self.num_segments)
                offsets += start_idx
            else:
                offsets = over_sample_strategy(total_len)
            # print(f'dense offsets: {offsets}')
            return offsets

        def non_dense_sample(total_len):
            average_duration = total_len // self.num_segments
            if average_duration > 1:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=self.num_segments)
            else:
                offsets = over_sample_strategy(total_len)
            return offsets

        if self.dense_sample:
            if random.random() < 0.5:
                offsets = dense_sample(base_idxs_len)
            else:
                offsets = non_dense_sample(base_idxs_len)
        else:
            offsets = non_dense_sample(base_idxs_len)

        return base_idxs[offsets].tolist(), real_idxs

    def sample_indices_test(self, video_len, data_list):
        """Frame sampling strategy in test stage.

        Args:
            video_len (int): Video frame count.

        """
        base_idxs = np.array(range(video_len), np.int)
        count = 0
        for i, item in enumerate(data_list):
            if not item or len(item) == 0:
                # print("=====================")
                base_idxs = np.delete(base_idxs, count)
                continue
            count += 1
        if self.sparse_span:
            # base_idxs = np.linspace(0, video_len - 1, self.sparse_span, dtype=np.int)
            base_idxs_index = np.linspace(0, len(base_idxs) - 1, self.sparse_span, dtype=np.int)
            base_idxs = base_idxs[base_idxs_index]
        base_idxs_len = len(base_idxs)

        if self.dense_sample:
            start_idx = max(base_idxs_len // 2 - self.dense_sample // 2, 0)
            end_idx = min(base_idxs_len // 2 + self.dense_sample // 2, base_idxs_len)
            base_idxs = base_idxs[start_idx: end_idx]
            base_idxs_len = len(base_idxs)

        tick = base_idxs_len / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        offsets = base_idxs[offsets].tolist()

        return offsets

    def get_enclosing_box(self, img_h, img_w, box, margin):
        """Get the square-shape face bounding box after enlarging by a certain margin.

        Args:
            img_h (int): Image height.
            img_w (int): Image width.
            box (list): [x0, y0, x1, y1] format face bounding box.
            margin (int): The margin to enlarge.

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

    def decode_selected_frames(self, vr, sampled_idxs, video_face_info_d):
        """Decode image frames from a given video on the fly.

        Args:
            vr (object): 
                Decord VideoReader instance.
            sampled_idxs (list): 
                List containing the frames to extract from the given video.
            video_face_info_d (dict): 
                Dict containing the face bounding box information of each frame from the given video.
        """
        img_h, img_w, _ = vr[0].shape
        frames = vr.get_batch(sampled_idxs).asnumpy()

        if self.split == 'train':
            margin = random.uniform(1.0, 1.5)
        else:
            margin = self.test_margin

        imgs = []
        for idx in range(len(frames)):
            try:
                img = frames[idx]
                x0, y0, x1, y1 = self.get_enclosing_box(img_h, img_w, video_face_info_d[sampled_idxs[idx]], margin)
                img = img[y0:y1, x0:x1]
                imgs.append(img)
            except:
                print("no face pkl .....")
        return imgs

    # 509.mp4
    def __getitem__(self, index):
        video_name, video_label = self.dataset_info[index]
        # ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        # video_path = os.path.join(eval(f'self.{video_label}_video_dir'), video_name + '.mp4')
        if video_label == "real":
            video_path = f"/mnt/e/DeepFakeDetection/datasets/FF++/original_sequences/youtube/c23/videos/{video_name}.mp4"
        elif video_label in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            video_path = f"/mnt/e/DeepFakeDetection/datasets/FF++/manipulated_sequences/{video_label}/c23/videos/{video_name}.mp4"
        elif video_label == "dfdc_real":
            pass
        elif video_label == "dfdc_fake":
            pass
        else:
            raise
        video_face_info_d = self.face_info[video_name.split('_')[0]]
        vr = VideoReader(video_path)
        video_len = min(len(vr), len(video_face_info_d))
        if video_name == "509":
            print("可算逮到你了!")
        if self.split == 'train':
            sampled_idxs, real_idxs = self.sample_indices_train(video_len, video_face_info_d[:video_len])
        else:
            sampled_idxs = self.sample_indices_test(video_len, video_face_info_d[:video_len])
        # frames list, 每个元素是人脸框rgb 矩阵
        # sampled_idxs_new = []
        # temp2 = []
        # for i in sampled_idxs:
        #     if i - 2 < 0:
        #         i = 2
        #     if i + 2 > video_len - 1:
        #         i = video_len - 3
        #
        #     sampled_idxs_new.extend([i - 2, i - 1, i, i + 1, i + 2])
        #     temp2.extend([real_idxs[i - 2], real_idxs[i - 1], real_idxs[i], real_idxs[i + 1], real_idxs[i + 2]])
        frames = self.decode_selected_frames(vr, sampled_idxs, video_face_info_d)
        # for i, frame in enumerate(frames):
        #     cv2.imwrite(f"test_imgs/{i}.jpg", frame[:, :, ::-1])
        if self.transform is not None:

            if random.random() < 0.5:
                frames = frames[::-1]

            # make sure the augmentation parameter is applied the same on each frame.
            additional_targets = {}
            tmp_imgs = {"image": frames[0]}
            for i in range(1, len(frames)):
                additional_targets[f"image{i}"] = "image"
                tmp_imgs[f"image{i}"] = frames[i]
            self.transform.add_targets(additional_targets)

            frames = self.transform(**tmp_imgs)
            frames = OrderedDict(sorted(frames.items(), key=lambda x: x[0]))
            frames = list(frames.values())
            frames = torch.stack(frames)  # T, C, H, W
            process_imgs = frames.view(-1, frames.size(2), frames.size(3)).contiguous()  # TC, H, W
        else:
            process_imgs = frames

        video_label_int = 0 if video_label == 'real' else 1
        torch.save()
        return process_imgs, video_label_int, video_path, sampled_idxs

    def __len__(self):
        return len(self.dataset_info)
