import os

# 定义要处理的根目录
root_dir = os.path.dirname(os.path.abspath(__file__))

# 定义需要替换的错误前缀
replacements = {
    "from Ghypeddings.HGCAE.": "from ",
    "from Ghypeddings.": "from ",
    "import Ghypeddings.": "import "
}


def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已修复: {file_path}")


# 遍历所有 Python 文件
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py") and file != "fix_paths.py":
            fix_file(os.path.join(subdir, file))

print("\n✨ 项目内所有绝对路径引用已清理完毕！")