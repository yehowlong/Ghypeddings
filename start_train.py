import os
import sys
import torch
import numpy as np

# 1. 路径环境注入，确保各模块能相互找到
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 正在初始化 HGCAE 训练环境...")

try:
    # 2. 导入数据集类
    # 注意：根据 datasets.py 源码，类名是 Cora (虽然它继承自 Dataset)
    from datasets.datasets import Cora

    # 3. 准备数据
    # 如果是第一次运行，build() 会读取 cora.content/cites 并生成 pkl 文件
    dataset_manager = Cora()
    print("📂 正在加载/构建 Cora 数据集...")
    adj, features, labels = dataset_manager.build()

    # 4. 实例化模型Runner
    # 你的 hgcae.py 已经处理了 process_data 和参数创建
    from HGCAE.hgcae import HGCAE

    print("🏗️ 正在初始化模型架构...")
    model_runner = HGCAE(
        adj=adj,
        features=features,
        labels=labels,
        dim=16,  # 嵌入维度
        hidden_dim=16,  # 隐藏层维度
        epochs=100,  # 训练轮数
        cuda=-1  # 使用 CPU (如果有GPU可改为0)
    )

    print("-" * 30)
    print("📈 开始训练 (Calling fit)...")

    # 5. 执行训练
    # 你的 hgcae.py 中定义的是 fit() 而不是 train()
    history, elapsed_time = model_runner.fit()

    print("-" * 30)
    print(f"✅ 训练完成！耗时: {elapsed_time:.2f}s")
    print(f"📊 最终训练损失: {history['train'][-1]:.4f}")

    # 6. 保存嵌入 (可选)
    # model_runner.save_embeddings(current_dir)

except Exception as e:
    print(f"❌ 运行失败: {e}")
    import traceback

    traceback.print_exc()