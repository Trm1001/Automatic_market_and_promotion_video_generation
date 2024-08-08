import subprocess

# 构建命令
command = "python sample_scripts/sample.py --config sample_scripts/configs/panda.yaml"

process = subprocess.run(command, shell=True, text=True, capture_output=True)
print("STDOUT:", process.stdout)
print("STDERR:", process.stderr)

