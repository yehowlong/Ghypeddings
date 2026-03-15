import sys
import os

# 获取当前脚本所在目录（即 Ghypeddings 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录（即 PycharmProjects 文件夹）
parent_dir = os.path.dirname(current_dir)

# 将父目录加入路径，这样 "import Ghypeddings" 就能生效
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 现在模拟命令行运行 HGCAE.hgcae
from HGCAE.hgcae import main
if __name__ == "__main__":
    # 这里可以手动指定参数，或者让它接收命令行参数
    import sys
    # 模拟命令行参数，你可以根据需要修改
    sys.argv = ['hgcae.py', '--dataset', 'Cora', '--dim', '10']
    main()