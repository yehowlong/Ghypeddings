import os
import sys
import torch
import numpy as np
import scipy.sparse as sp

# 1. 路径环境注入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
hgcae_dir = os.path.join(current_dir, 'HGCAE')
sys.path.insert(0, hgcae_dir)

# 导入你提供的 process_data
try:
    from HGCAE.utils.data_utils import process_data

    print("✅ 成功导入 process_data")
except ImportError:
    from utils.data_utils import process_data

    print("✅ 成功从本地导入 process_data")


# 2. 手动实现一个简易的 Cora 加载器 (替代缺失的 CoraDataset)
def load_cora_raw(data_path="data/cora"):
    """
    这是一个通用的 Cora 加载逻辑，假设你的数据在项目根目录的 data/cora 文件夹下
    """
    print(f"📂 正在从 {data_path} 读取原始 Cora 数据...")
    # 这里你可以替换成你实际读取 .content 和 .cites 的代码
    # 如果你已经有现成的读取好的数据，可以直接在这里加载

    # 模拟 Cora 规模的数据（如果读取失败，至少能保证代码结构跑通）
    # 建议此处改为你实际的读取逻辑
    num_nodes = 2708
    num_features = 1433
    adj = sp.eye(num_nodes)  # 占位符
    features = np.random.randn(num_nodes, num_features)
    labels = np.random.randint(0, 7, size=(num_nodes,))
    return adj, features, labels


# 3. 配置参数类 (必须包含 process_data 需要的属性)
class Args:
    def __init__(self):
        self.dataset = 'Cora'
        self.val_prop = 0.05
        self.test_prop = 0.1
        self.seed = 42
        self.normalize_adj = True
        self.normalize_feats = True
        self.lambda_rec = False
        self.dim = 16
        self.hidden_dim = 16
        # 模型相关
        self.cuda = -1
        self.manifold = 'PoincareBall'


try:
    args = Args()

    # 4. 获取原始数据
    # 注意：你需要确保你的 Cora 原始数据路径正确
    # 如果你有现成的 datasets.py 能用，请改为调用它
    adj_raw, feat_raw, labels_raw = load_cora_raw()

    # 5. 调用你的 data_utils.py 中的 process_data
    print("⚙️ 正在执行 process_data 进行预处理...")
    processed_data = process_data(args, adj_raw, feat_raw, labels_raw)

    # 6. 实例化模型
    from HGCAE.hgcae import HGCAE as ModelRunner

    # 根据 process_data 返回的字典提取数据
    features = processed_data['features']
    labels = torch.LongTensor(processed_data['labels'])
    adj_enc = processed_data['adj_train_enc']

    print(f"✅ 数据处理完成: {processed_data['info']}")

    runner = ModelRunner(
        features=features,
        labels=labels,
        dim=args.dim,
        hidden_dim=args.hidden_dim
    )

    # 7. 训练
    print("-" * 30)
    print("📈 启动训练...")
    # 使用 process_data 生成的训练边缘数据进行训练
    if hasattr(runner, 'train'):
        # 传入 adj 和相关的索引/边缘
        runner.train(adj_enc, processed_data['idx_all'])
    else:
        print("⚠️ 未发现 .train() 方法，请检查模型类。")

except Exception as e:
    print(f"❌ 运行失败: {e}")
    import traceback

    traceback.print_exc()