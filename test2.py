from moviepy.editor import VideoFileClip, concatenate_videoclips

# 加载视频文件
video1 = VideoFileClip("./generate_results/5/sampled_video.mp4")
video2 = VideoFileClip("./generate_results/1/sampled_video.mp4")
video3 = VideoFileClip("./generate_results/3/sampled_video.mp4")
video4 = VideoFileClip("./generate_results/4/sampled_video.mp4")

# 拼接视频
final_video = concatenate_videoclips([video1, video2, video3, video4])

# 导出拼接后的视频
final_video.write_videofile("output.mp4", codec="libx264", fps=24)
