import os
import sys

# ================= 配置区域 =================
TARGET_DIR = "."          # 目标目录，"." 表示当前目录
OLD_STR = "GradPrint"         # 要替换的旧字符串
NEW_STR = "GradPrint"     # 新字符串
# 要忽略的目录或文件后缀（防止损坏二进制文件或版本控制）
IGNORE_DIRS = {'.git', '.idea', '__pycache__', 'venv', 'env'}
IGNORE_EXTS = {'.pyc', '.pyd', '.so', '.dll', '.exe', '.git', 
               '.pdf', '.png', '.jpg', '.jpeg', '.zip', '.tar', '.gz', 
               '.pkl', '.pth', '.model', '.h5'} # 模型文件和图片不要改内容
# ===========================================

def is_text_file(filepath):
    """
    简单的启发式检查：尝试以 UTF-8 读取文件前 1024 字节。
    如果失败，则认为是二进制文件，不进行内容替换。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

def replace_content(root_dir):
    """步骤 1: 替换文件内容"""
    print(f"[*] Step 1: Replacing content ('{OLD_STR}' -> '{NEW_STR}')...")
    count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # 过滤忽略的目录
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in IGNORE_EXTS:
                continue

            file_path = os.path.join(root, file)
            
            # 检查是否为文本文件
            if not is_text_file(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if OLD_STR in content:
                    new_content = content.replace(OLD_STR, NEW_STR)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"    Modified content: {file_path}")
                    count += 1
            except Exception as e:
                print(f"    Error reading {file_path}: {e}")

    print(f"[*] Content replacement finished. {count} files modified.\n")

def rename_files_and_dirs(root_dir):
    """步骤 2: 重命名文件和文件夹 (自底向上遍历)"""
    print(f"[*] Step 2: Renaming files and directories...")
    rename_count = 0

    # topdown=False 是关键！必须先重命名子文件/子文件夹，再重命名父文件夹，
    # 否则路径会失效。
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # 过滤忽略的目录（虽然 topdown=False，但为了安全还是检查路径）
        if any(ignore in root.split(os.sep) for ignore in IGNORE_DIRS):
            continue

        # 1. 重命名文件
        for filename in files:
            if OLD_STR in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(OLD_STR, NEW_STR)
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"    Renamed File: {filename} -> {new_filename}")
                    rename_count += 1
                except Exception as e:
                    print(f"    Error renaming file {old_path}: {e}")

        # 2. 重命名文件夹
        for dirname in dirs:
            if OLD_STR in dirname:
                old_path = os.path.join(root, dirname)
                new_dirname = dirname.replace(OLD_STR, NEW_STR)
                new_path = os.path.join(root, new_dirname)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"    Renamed Dir : {dirname} -> {new_dirname}")
                    rename_count += 1
                except Exception as e:
                    print(f"    Error renaming dir {old_path}: {e}")

    print(f"[*] Renaming finished. {rename_count} items renamed.\n")

if __name__ == "__main__":
    print("========================================================")
    print(f"⚠️  WARNING: This script will modify files in: {os.path.abspath(TARGET_DIR)}")
    print(f"⚠️  Target: Replace '{OLD_STR}' with '{NEW_STR}'")
    print("⚠️  PLEASE MAKE SURE YOU HAVE A BACKUP!")
    print("========================================================")
    
    confirm = input("Type 'yes' to continue: ")
    if confirm.lower() == 'yes':
        replace_content(TARGET_DIR)
        rename_files_and_dirs(TARGET_DIR)
        print("✅ All Done! Please check your imports and run tests.")
    else:
        print("Cancelled.")