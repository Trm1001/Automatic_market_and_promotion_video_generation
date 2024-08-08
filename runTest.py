import subprocess

# 定义训练脚本路径和参数
script_path = "TestLoader.py"
model = "TAVU"
num_frames = 16
dataset = "WebVideoImageStage1"
frame_interval = 4
ckpt_every = 5
clip_max_norm = 0.1
global_batch_size = 2
reg_text_weight = 0
results_dir = "./results"
pretrained_t2v_model = "pretrained/watermark_remove_module.pt"
global_mapper_path = "pretrained/mapper.pt"
data_path = "path-to-videobooth-subset/"

# 构建命令
command = [
    "torchrun",
    "--nnodes=1",
    "--nproc_per_node=1",
    "--master_port=29125",
    script_path,
    "--model", model,
    "--num-frames", str(num_frames),
    "--dataset", dataset,
    "--frame-interval", str(frame_interval),
    "--ckpt-every", str(ckpt_every),
    "--clip-max-norm", str(clip_max_norm),
    "--global-batch-size", str(global_batch_size),
    "--reg-text-weight", str(reg_text_weight),
    "--results-dir", results_dir,
    "--pretrained-t2v-model", pretrained_t2v_model,
    "--global-mapper-path", global_mapper_path,
    "--data_path",data_path
]

# 运行命令
subprocess.run(command)
