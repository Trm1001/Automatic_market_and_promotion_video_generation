from torchvision import transforms
from datasets import video_transforms


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