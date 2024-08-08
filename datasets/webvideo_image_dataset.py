import io
import os
import torch
import pandas
import decord
import random
import json

import numpy as np

import torchvision.transforms.functional as F
from datasets import video_transforms

# import albumentations as A
from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

class WebVideoImageStage1(torch.utils.data.Dataset):
    """Load the WebVideo video files

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_data = self.get_videodata(configs.data_path)
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()

        self.img_random_trans = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0, 0.1),
                scale=(0.8, 1.2),
                fill=255)
        ])

        self.parsing_dir = 'path-to-webvid-parsing'

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if normalize:
            transform_list += [transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        # start_time = time.perf_counter()

        max_retries = 3  # 最多重试3次
        retry_count = 0
        word_prompt = "unknown"  # 设置一个默认值，以防没有检测到任何目标

        while retry_count < max_retries:
            try:
                # load video data
                video_info = self.video_data.iloc[index]
                video_id, video_page_dir, video_name, mask_dir = video_info['videoid'], video_info['page_dir'], video_info['name'], video_info['mask_dir']
                video_path = 'path-to-webvid10M/{}/{}.mp4'.format(video_page_dir, video_id)
                print(f"Loading video from path: {video_path}")
                v_reader = self.v_decoder(video_path)

                total_frames = len(v_reader)

                # Sampling video frames
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len
                frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
                video = torch.from_numpy(v_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()

                # load image prompts
                parsing_mask_json_path  = f'{mask_dir}/{video_id}/mask.json'
                print(parsing_mask_json_path)

                with open(parsing_mask_json_path) as f:
                    label_list = json.load(f)

                target_label = random.choice(label_list)
                print(target_label)
                mask_path = f'{self.parsing_dir}/{video_id}/mask_1.png'

                mask = np.array(Image.open(mask_path))
                first_frame = v_reader.get_batch([0]).asnumpy()[0]

                middle_frame_index = total_frames // 2  # 计算中间帧的索引
                middle_frame = v_reader.get_batch([middle_frame_index]).asnumpy()[0]
                masked_first_frame = middle_frame.copy()
                masked_first_frame[mask==0] = 255
                print("走到这一步了1")
                print(target_label['detections'])
                print(type(target_label), target_label)

                print("走到这一步了1.5")

                detections = label_list[0]['detections']

                print(detections)
                print("走到这一步了1.6")

                if len(detections) > 0:  # 检查detections列表是否非空
                    print("走到这一步了1.7")

                    target_label = random.choice(detections)  # 从 detections 列表中随机选择一个元素
                    print("走到这一步了2")
                    word_prompt = target_label['label']  # 获取标签

                    print("走到这一步了3")
                    print(word_prompt)

                    x1, y1, w, h = target_label['box']
                    x2, y2 = x1 + w, y1 + h  # 计算结束坐标

                    # 接下来使用 word_prompt 和框 (x1, y1, x2, y2)
                else:
                    print(f"No detections for frame_id {target_label['frame_id']}")  # 如果没有检测到任何对象
                    masked_first_frame = torch.zeros([3, 224, 224])  # 创建一个空的masked_first_frame作为占位符

                print("走到6")
                masked_first_frame = masked_first_frame[int(y1):int(y2), int(x1):int(x2), :]
                print(f"y1:{y1},y2:{y2},x1:{x1},x2:{x2}")

                # 检查 masked_first_frame 是否是 np.ndarray，如果是，则转换为 Tensor
                if isinstance(masked_first_frame, np.ndarray):
                    masked_first_frame = torch.from_numpy(masked_first_frame).permute(2, 0, 1).contiguous()
                else:
                    # 如果已经是 Tensor，确保其维度顺序正确
                    if masked_first_frame.dim() == 3 and masked_first_frame.shape[0] != 3:
                        masked_first_frame = masked_first_frame.permute(2, 0, 1).contiguous()

                # 现在 masked_first_frame 是一个正确形状的 Tensor，可以继续进行后续处理

                print("走到7")
                height, width = masked_first_frame.size(1), masked_first_frame.size(2)
                print(f"height:{height},width:{width}")
                print("走到8")

                if height == width:
                    pass
                elif height < width:
                    diff = width - height
                    top_pad = diff // 2
                    down_pad = diff - top_pad
                    left_pad = 0
                    right_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)
                else:
                    diff = height - width
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    top_pad = 0
                    down_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)
                print("走到12345")
                masked_first_frame = self.img_random_trans(masked_first_frame)
                print("走到9")
                masked_first_frame = masked_first_frame / 255.0
                print("走到10")
                # masked_first_frame = self.get_tensor_clip()(masked_first_frame)

                del v_reader
                print("12345")
                break
            # except Exception as e:
            #     print(e)
            #     index = random.randint(0, len(self.video_data) - 1)
            except Exception as e:
                print(f"Error processing video: {e}")
                retry_count += 1  # 增加重试计数
                if retry_count >= max_retries:
                    print("Failed after maximum retries. Skipping this entry.")
                    return None  # 返回None或者默认数据

        if retry_count == max_retries:
            print("Failed after maximum retries. Skipping this entry.")
        # videotransformer data proprecess
        video = self.transform(video) # T C H W
        print(f"Successfully processed video: {video_path}")
        return {'video': video, 'video_name': video_name, 'word_prompt': word_prompt, 'masked_first_frame': masked_first_frame}

    def __len__(self):
        return len(self.video_data)

    def get_videodata(self, datainfo_path, split='train'):

        target_animal_list = ['human']
        pandas_frame_list = []
        for target_animal in target_animal_list:
            parsing_dir = f'path-to-videobooth-subset/{target_animal}'
            datainfo_path = f'path-to-videobooth-subset/{target_animal}.csv'
            animal_video_data = pandas.read_csv(datainfo_path,  usecols=[3, 4, 6])
            animal_video_data['mask_dir'] = [parsing_dir] * len(animal_video_data)
            pandas_frame_list.append(animal_video_data)

        video_data = pandas.concat(pandas_frame_list)

        return video_data


class WebVideoImageStage2(torch.utils.data.Dataset):
    """Load the WebVideo video files

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_data = self.get_videodata(configs.data_path)
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()

        self.img_random_trans = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0, 0.1),
                scale=(0.8, 1.2),
                fill=255)
        ])

        self.first_frame_random_trans = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                fill=255)
        ])

        self.parsing_dir = 'path-to-webvid-parsing'

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if normalize:
            transform_list += [transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):

        while True:
            try:
                # load video data
                video_info = self.video_data.iloc[index]
                video_id, video_page_dir, video_name, mask_dir = video_info['videoid'], video_info['page_dir'], video_info['name'], video_info['mask_dir']
                video_path = 'path-to-webvid10M/{}/{}.mp4'.format(video_page_dir, video_id)

                v_reader = self.v_decoder(video_path)

                total_frames = len(v_reader)

                # Sampling video frames
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len
                frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
                video = torch.from_numpy(v_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()

                # load image prompts
                parsing_mask_json_path  = f'{mask_dir}/{video_id}/mask.json'

                with open(parsing_mask_json_path) as f:
                    label_list = json.load(f)

                target_label = random.choice(label_list)
                mask_path = f'{self.parsing_dir}/{video_id}/mask_{target_label["value"]}.png'

                mask = np.array(Image.open(mask_path))
                first_frame = v_reader.get_batch([0]).asnumpy()[0]

                masked_first_frame = first_frame.copy()
                masked_first_frame[mask==0] = 255

                word_prompt = target_label['label']

                x1, y1, x2, y2 = target_label['box']

                # random crop the bbox
                augmentation_type = random.uniform(0, 1)
                if augmentation_type < 0.25:
                    y1 = y1 + random.uniform(0.01, 0.2) * (y2 - y1)
                elif augmentation_type < 0.50:
                    y2 = y2 - random.uniform(0.01, 0.2) * (y2 - y1)
                elif augmentation_type < 0.75:
                    x1 = x1 + random.uniform(0.01, 0.2) * (x2 - x1)
                elif augmentation_type < 1.0:
                    x2 = x2 - random.uniform(0.01, 0.2) * (x2 - x1)

                masked_first_frame = masked_first_frame[int(y1):int(y2), int(x1):int(x2), :]

                masked_first_frame = torch.from_numpy(masked_first_frame).permute(2, 0, 1).contiguous()
                height, width = masked_first_frame.size(1), masked_first_frame.size(2)

                if height == width:
                    pass
                elif height < width:
                    diff = width - height
                    top_pad = diff // 2
                    down_pad = diff - top_pad
                    left_pad = 0
                    right_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)
                else:
                    diff = height - width
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    top_pad = 0
                    down_pad = 0
                    padding_size = [left_pad, top_pad, right_pad, down_pad]
                    masked_first_frame = F.pad(masked_first_frame, padding=padding_size, fill = 255)

                masked_first_frame_ori = masked_first_frame.clone()
                masked_first_frame = self.img_random_trans(masked_first_frame)
                masked_first_frame = masked_first_frame / 255.0
                # masked_first_frame = self.get_tensor_clip()(masked_first_frame)

                # video_with_first_frame = torch.cat([self.first_frame_random_trans(torch.from_numpy(masked_first_frame_original).permute(2, 0, 1).contiguous().unsqueeze(0)), video], dim=0)
                aug_first_frame = self.first_frame_random_trans(masked_first_frame_ori).unsqueeze(0) / 127.5 - 1

                del v_reader
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.video_data) - 1)

        # videotransformer data proprecess
        video = self.transform(video) # T C H W
        video = torch.cat([aug_first_frame, video], dim = 0)
        print("走到这一步了5")
        return {'video': video, 'video_name': video_name, 'word_prompt': word_prompt, 'masked_first_frame': masked_first_frame, 'index': f'{index}'}

    def __len__(self):
        return len(self.video_data)

    def get_videodata(self, datainfo_path, split='train'):

        target_animal_list = ['dog', 'cat', 'bear', 'car', 'panda', 'tiger', 'horse', 'elephant', 'lion']
        pandas_frame_list = []
        for target_animal in target_animal_list:
            parsing_dir = f'path-to-videobooth-subset/{target_animal}'
            datainfo_path = f'path-to-videobooth-subset/{target_animal}.csv'
            animal_video_data = pandas.read_csv(datainfo_path,  usecols=[3, 4, 6])
            animal_video_data['mask_dir'] = [parsing_dir] * len(animal_video_data)
            pandas_frame_list.append(animal_video_data)

        video_data = pandas.concat(pandas_frame_list)

        return video_data

