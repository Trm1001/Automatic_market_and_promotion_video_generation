import subprocess

# 定义配置参数
img_path = "./sample_scripts/samples/images/mask_1.jpg"
mask_path = "./sample_scripts/samples/segments/panda/mask_1_video_prev_ui_gray_image.png"
text_prompt = "Human standing behind a podium and giving a lecture"
replace_word = "human"
seed = 11
bbox = [3.7753829956054688, 19.906082153320312, 221.86465454101562, 162.71131896972656]

# 构建命令
command = [
    "python",
    "sample_scripts/sample.py",
    "--config", "sample_scripts/configs/panda.yaml"
]

# 运行命令
result = subprocess.run(command, capture_output=True, text=True)

# 输出调用结果
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")

