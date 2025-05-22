import os
import pandas as pd

def merge_csv_files(base_directory, output_path):
    """
    合并指定路径下的所有CSV文件为一个CSV文件，并移除重复记录

    :param base_directory: 包含文件夹的根目录路径
    :param output_path: 合并后的CSV文件保存路径
    """
    # 初始化存储数据的列表
    merged_data = []
    total_rows = 0
    
    # 遍历base_directory及其子目录中的所有文件
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".csv"):
                # 构建CSV文件的完整路径
                csv_file_path = os.path.join(root, file)
                print(f"正在处理文件: {csv_file_path}")

                # 读取CSV文件
                try:
                    data = pd.read_csv(csv_file_path)
                    total_rows += len(data)
                    merged_data.append(data)
                except Exception as e:
                    print(f"无法读取文件 {csv_file_path}: {e}")

    # 如果没有找到任何CSV文件
    if not merged_data:
        print("没有找到任何CSV文件！")
        return

    # 合并所有的DataFrame
    print("正在合并数据...")
    merged_df = pd.concat(merged_data, ignore_index=True)
    print(f"合并前总行数: {total_rows}")
    
    # 去除完全重复的行
    print("正在移除重复数据...")
    merged_df_no_duplicates = merged_df.drop_duplicates()
    duplicates_count = len(merged_df) - len(merged_df_no_duplicates)
    print(f"发现并移除了 {duplicates_count} 行重复数据")
    print(f"去重后总行数: {len(merged_df_no_duplicates)}")

    # 保存合并后的CSV文件
    try:
        merged_df_no_duplicates.to_csv(output_path, index=False)
        print(f"合并后的CSV文件已保存到: {output_path}")
    except Exception as e:
        print(f"保存合并文件时出错: {e}")

# 调用函数
if __name__ == "__main__":
    base_directory = r"M:\MATCH_RESULT"  # 输入文件夹路径
    output_path = r"M:\Data_Set.csv"  # 输出文件路径
    merge_csv_files(base_directory, output_path)