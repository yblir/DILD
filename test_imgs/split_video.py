# -*- coding: utf-8 -*-
# @Time    : 2023/5/23 15:06
# @Author  : yblir
# @File    : split_video.py
# explain  : 
# =======================================================
import os
import json
from pathlib2 import Path

def change_mp4file(file_path):
    json_file = file_path + '/metadata.json'
    with open(json_file, 'r') as f:
        # info = json.loads(f.read())
        info = json.load(f)

    for file in [os.path.join(file_path, i) for i in os.listdir(file_path)]:
        if file.endswith('.mp4'):
            print('====>Fake file', file)
            mp4_file = file.split('.')[-2].split('/')[-1] + '.mp4'
            # print('----Create Fake Video-->',mp4_file)
            label = info[mp4_file]['label']
            if label == 'FAKE':
                original = info[mp4_file]['original'].split('.')[0] + '.mp4'
                print('===Original real Video===>', original)
                new_file_name = file.split('.')[-2] + '_' + original
                print('===New Deepfake video==>', new_file_name)
                os.rename(file, new_file_name)


def mp4_json(file_path):
    json_list = []
    for file in [os.path.join(file_path, i) for i in os.listdir(file_path)]:
        if file.endswith('.mp4'):
            mp4_name = file.split('/')[-1].split('.')[0]
            if len(mp4_name.split('_')) == 2:
                # print('===>',mp4_name)
                name1 = mp4_name.split('_')[0]
                name2 = mp4_name.split('_')[1]
                name_list = [name1, name2]
                print('=====>', name_list)
                json_list.append(name_list)
    with open('train.json', 'w') as f:
        f.write(json.dumps(json_list))


if __name__ == '__main__':
    file_path = '/mnt/d/Downloads/dfdc_train_part_00/dfdc_train_part_0'
    change_mp4file(file_path)
    mp4_json(file_path)
