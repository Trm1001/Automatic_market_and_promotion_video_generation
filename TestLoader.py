import torch
from torch.utils.data import DataLoader, DistributedSampler
import logging
import argparse
import torch.distributed as dist
from datasets import video_transforms
from torchvision import transforms
import itertools

def get_dataset(args):
    from datasets import webvideo_image_dataset
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'WebVideoImageStage1':
        transform_webvideo = transforms.Compose([
            video_transforms.ToTensorVideo(),
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.CenterCropResizeVideo(args.image_size), # center crop using shor edge, then resize
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        return webvideo_image_dataset.WebVideoImageStage1(args, transform=transform_webvideo, temporal_sample=temporal_sample)
    elif args.dataset == 'WebVideoImageStage2':
        transform_webvideo = transforms.Compose([
            video_transforms.ToTensorVideo(),
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.CenterCropResizeVideo(args.image_size), # center crop using shor edge, then resize
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        return webvideo_image_dataset.WebVideoImageStage2(args, transform=transform_webvideo, temporal_sample=temporal_sample)
    else:
        raise NotImplementedError(args.dataset)


# 假设 args 是通过 argparse 获取的，您可以根据需要调整或硬编码某些参数
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
parser.add_argument("--num-classes", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--global-batch-size", type=int, default=256)
parser.add_argument("--global-seed", type=int, default=3407)
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--ckpt-every", type=int, default=20000)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--global_seed', type=int, default=42)
# added by maxin
parser.add_argument("--class-guided", default=False, action='store_true')
parser.add_argument("--use-timecross-transformer", default=False, action='store_true')
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--num-frames", type=int, default=16, help='video frames for training')
parser.add_argument("--frame-interval", type=int, default=1, help='video frames interval')
parser.add_argument("--attention-mode", default='math', type=str, help='which attention used')
parser.add_argument("--dataset", type=str, default='ffs', help='dataset for training')
parser.add_argument("--clip-max-norm", default=None, type=float, help='clip gradient')
parser.add_argument("--use-compile", default=False, action='store_true', help='speedup by torch compile')
parser.add_argument("--global-mapper-path", type=str, default=None)
parser.add_argument("--num-sampling-steps", type=int, default=250)
parser.add_argument("--reg-weight", type=float, default=0.01)
parser.add_argument("--reg-text-weight", type=float, default=0.01)
parser.add_argument("--cfg-scale", type=float, default=4.0)
parser.add_argument("--pretrained-t2v-model", type=str, required=True, default='pretrained/watermark_remove_module.pt')
parser.add_argument('--model', type=str, default='TAVU', required=True, help='Model type')  # 添加这个参数
parser.add_argument('--log-file', type=str, default='train.log', help='Log file path')
parser.add_argument('--data_path', type=str, default='path-to-videobooth-subset/', required=True, help='Path to the dataset')
args = parser.parse_args()

# 初始化分布式环境（如果已经在使用，如果不是，需要调整或移除相关代码）
dist.init_process_group(backend='nccl')
rank = dist.get_rank()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# 设置数据
dataset = get_dataset(args)
logger.info(f"Dataset contains {len(dataset):,} videos.")

try:
    for i, data in enumerate(itertools.islice(dataset, len(dataset))):
        logger.info(f"Processing item {i+1}/{len(dataset)}")
        # Here you would typically process your data
        print("Data:", data)  # Replace or augment this with your actual data handling code

        # If 'data' has a method or attribute that can be displayed or checked
        # For example, if data is a video frame or some object with methods
        if hasattr(data, 'shape'):
            print("Data shape:", data.shape)
        if hasattr(data, '__len__'):
            print("Number of elements in data:", len(data))

except Exception as e:
    logger.error("An error occurred during dataset processing", exc_info=True)
    print("Error:", e)

logger.info("Finished processing dataset.")

# 尝试直接访问几个数据项
for i, data in enumerate(itertools.islice(dataset, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}: {sample}")  # 打印输出样本查看是否正常
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=args.global_seed
)

loader = DataLoader(
    dataset,
    batch_size=int(args.global_batch_size // dist.get_world_size()),
    shuffle=False,
    sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)

logger.info(f"Dataset contains {len(dataset):,} videos.")

# 测试数据加载
try:
    for i, data in enumerate(loader):
        logger.info(f"Loaded batch {i+1}, batch size: {len(data)}")  # 假设 data 是标准输出格式
        if i == 1:  # 只加载两个批次来测试
            break
except Exception as e:
    logger.error(f"Error during data loading: {str(e)}")
