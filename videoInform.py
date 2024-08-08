import csv
import os
import cv2  # 用于读取视频信息

# 假设你的视频存储在这个目录
video_directory = 'path-to-videobooth-subset/human/'

# CSV 文件保存位置
csv_file = 'human.csv'

# 开始创建 CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入列标题
    writer.writerow(['videoid', 'name', 'page_idx', 'page_dir', 'duration', 'contentUrl'])

    # 遍历视频目录
    for idx, video_filename in enumerate(os.listdir(video_directory)):
        if video_filename.endswith('.mp4'):  # 假设视频文件是 mp4 格式
            video_path = os.path.join(video_directory, video_filename)
            cap = cv2.VideoCapture(video_path)
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # 写入每行数据
            writer.writerow([idx, video_filename, idx, 'your_page_dir', duration, video_path])
