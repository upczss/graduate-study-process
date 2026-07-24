import torch
import sys

print("=" * 50)
print(f"当前Python解释器版本：{sys.version}")
print("=" * 50)
print(f"PyTorch 版本：{torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用：{cuda_available}")

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"显卡型号：{gpu_name}")
    print(f"PyTorch内置CUDA版本：{torch.version.cuda}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显卡总显存：{total_mem:.2f} GB")
else:
    print("未检测到可用NVIDIA显卡，仅能使用CPU运行")

print("=" * 50)
print("环境检测完毕！")