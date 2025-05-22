import os

# 设置数据路径
l1b_dir = r"M:\TROPOMI_S5P\temp"

def rename_files_in_directory(directory):
    """遍历目录中的所有文件，重命名符合条件的文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".zip"):  # 只处理.zip后缀的文件
                # 查找文件名中最后一个"_"字符的位置
                last_underscore_pos = file.rfind("_")
                if last_underscore_pos != -1:
                    # 截取文件名并去掉后缀
                    new_name = file[:last_underscore_pos] # + ".nc"
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_name)

                    try:
                        # 重命名文件
                        os.rename(old_file_path, new_file_path)
                        print(f"文件重命名成功: {old_file_path} -> {new_file_path}")
                    except Exception as e:
                        print(f"文件重命名失败: {old_file_path} -> {new_file_path} 错误: {str(e)}")

# 执行重命名操作
rename_files_in_directory(l1b_dir)
