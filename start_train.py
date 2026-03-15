import os
import sys
import torch
import numpy as np

# 1. 路径环境注入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 正在初始化 HGCAE 训练环境...")

try:
    # 2. 导入修复后的 Cora 类
    from datasets.datasets import Cora
    from HGCAE.hgcae import HGCAE

    # 3. 加载并构建数据
    # build() 会读取 cora.content 和 cora.cites
    dataset_manager = Cora()
    print("📂 正在解析 Cora 原始文件并构建图结构...")
    adj, features, labels = dataset_manager.build()

    # 4. 实例化 HGCAE
    # 根据 hgcae.py 源码，__init__ 会自动调用 process_data
    print("🏗️ 初始化双曲几何计算引擎...")
    runner = HGCAE(
        adj=adj,
        features=features,
        labels=labels,
        dim=16,
        hidden_dim=16,
        epochs=100,
        cuda=-1,  # 使用 CPU 运行
        dropout=0.1,
        lr=0.001
    )

    # 5. 开始训练
    print("-" * 30)
    print("📈 执行双曲空间嵌入学习 (fit)...")
    # fit() 会执行训练循环并返回损失历史
    history, elapsed_time = runner.fit()

    print("-" * 30)
    print(f"✅ 训练成功！耗时: {elapsed_time:.2f}s")
    print(f"📊 最终训练损失: {history['train'][-1]:.4f}")

except Exception as e:
    print(f"❌ 运行失败: {e}")
    import traceback

    traceback.print_exc()