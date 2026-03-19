import sys
import subprocess

def check(label, fn):
    try:
        result = fn()
        print(f"✅ {label}: {result}")
        return True
    except Exception as e:
        print(f"❌ {label}: {e}")
        return False

print("=" * 50)
print("环境检测报告")
print("=" * 50)

# Python 版本
check("Python 版本", lambda: sys.version.split()[0])

# PyTorch
def check_torch():
    import torch
    return f"{torch.__version__}"
check("PyTorch 版本", check_torch)

# CUDA 可用性
def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise Exception("CUDA 不可用")
    return f"可用，设备数量: {torch.cuda.device_count()}"
check("CUDA", check_cuda)

# GPU 型号和显存
def check_gpu():
    import torch
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}，显存: {mem:.1f} GB"
check("GPU 信息", check_gpu)

# CUDA 版本
def check_cuda_version():
    import torch
    return torch.version.cuda
check("CUDA 版本", check_cuda_version)

# transformers
def check_transformers():
    import transformers
    return transformers.__version__
check("transformers", check_transformers)

# trl（最关键）
def check_trl():
    import trl
    return trl.__version__
check("trl", check_trl)

# peft
def check_peft():
    import peft
    return peft.__version__
check("peft", check_peft)

# datasets
def check_datasets():
    import datasets
    return datasets.__version__
check("datasets", check_datasets)

# accelerate
def check_accelerate():
    import accelerate
    return accelerate.__version__
check("accelerate", check_accelerate)

# bitsandbytes（量化用，Windows 有坑）
def check_bnb():
    import bitsandbytes as bnb
    return bnb.__version__
check("bitsandbytes", check_bnb)

print("=" * 50)