import os
import sys
import subprocess

# 1. 获取关键路径
current_dir = os.path.dirname(os.path.abspath(__file__)) # Ghypeddings
hgcae_dir = os.path.join(current_dir, 'HGCAE')           # Ghypeddings/HGCAE

# 2. 设置环境变量 PYTHONPATH
# 这步最关键：它会告诉 Python 在运行 hgcae.py 时，
# 同时把 Ghypeddings 和 HGCAE 目录都当成搜索库的基地
env = os.environ.copy()
env["PYTHONPATH"] = f"{current_dir};{hgcae_dir};" + env.get("PYTHONPATH", "")

# 3. 构建运行命令
# 我们直接调用 python 运行 hgcae.py 文件，而不是通过 import
cmd = [
    sys.executable,
    os.path.join(hgcae_dir, 'hgcae.py'),
    '--dataset', 'Cora',
    '--dim', '10'
]

print("🚀 正在通过环境变量补丁启动 HGCAE 训练...")
print(f"当前工作目录: {current_dir}")

# 4. 执行
try:
    subprocess.run(cmd, env=env, check=True)
except subprocess.CalledProcessError as e:
    print(f"\n❌ 运行中途出错。这通常是由于代码内部还有绝对路径引用。")
except Exception as e:
    print(f"\n❌ 启动失败: {e}")