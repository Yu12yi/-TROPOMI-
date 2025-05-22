import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 根据实际文件结构，定义各变量在NetCDF文件中的完整路径
VARIABLE_PATHS = {
    "latitude": "PRODUCT/latitude",
    "longitude": "PRODUCT/longitude",
    "nitrogendioxide_tropospheric_column": "PRODUCT/nitrogendioxide_tropospheric_column",
    "nitrogendioxide_stratospheric_column": "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_stratospheric_column",
    "nitrogendioxide_total_column": "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_total_column"
}


def get_variable_value(dataset, var_path, file_path):
    """
    根据完整路径（例如 "PRODUCT/latitude" 或 "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_stratospheric_column"）
    逐级遍历分组，获取变量数据。
    """
    path_parts = var_path.split('/')
    current = dataset
    for part in path_parts[:-1]:
        found = False
        for group_name in current.groups:
            if group_name == part:
                current = current.groups[group_name]
                found = True
                break
        if not found:
            raise KeyError(f"Group '{part}' not found in file {file_path}")
    var_name_in_file = path_parts[-1]
    if var_name_in_file in current.variables:
        var_data = current.variables[var_name_in_file][:]
        # 若存在时间维度且仅有一个时间步，取第一个
        if var_data.ndim == 3 and var_data.shape[0] == 1:
            return var_data[0].astype(np.float32)
        else:
            return var_data.astype(np.float32)
    else:
        raise KeyError(f"变量 '{var_name_in_file}' not found in file {file_path}")


def read_tropomi_data(file_path):
    """
    读取TROPOMI L2级数据文件中所需变量的数据：
      - 经纬度和对流层NO₂位于PRODUCT/路径下；
      - 平流层NO₂及总NO₂位于PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/路径下；
    不进行QA过滤。
    """
    data = {}
    with nc.Dataset(file_path, 'r') as dataset:
        # 读取经纬度数据和对流层NO₂
        for var in ['latitude', 'longitude', 'nitrogendioxide_tropospheric_column']:
            var_path = VARIABLE_PATHS[var]
            data[var] = get_variable_value(dataset, var_path, file_path)
        # 读取平流层NO₂与总NO₂数据
        for var in ['nitrogendioxide_stratospheric_column', 'nitrogendioxide_total_column']:
            var_path = VARIABLE_PATHS[var]
            data[var] = get_variable_value(dataset, var_path, file_path)
    return data


def extract_date_from_filename(filename):
    """
    从文件名中提取日期编码：
    从第一个出现 "20" 开始，向后读取8位数字作为日期编码。
    例如：
    S5P_OFFL_L2__NO2____20230802T070747_20230802T084916_30062_03_020500_20230803T230157_clip.nc
    则提取日期编码为 "20230802"
    """
    idx = filename.find("20")
    if idx == -1 or len(filename) < idx + 8:
        raise ValueError("文件名格式异常，无法提取日期编码")
    return filename[idx:idx + 8]


def extract_no2_value(data, lat, lon):
    """
    对给定经纬度采用最近邻方法提取NO₂浓度：
      - 返回对流层、平流层和总NO₂浓度
    """
    lat_array = data['latitude']
    lon_array = data['longitude']
    # 展平数组并计算欧氏距离
    flat_lat = lat_array.flatten()
    flat_lon = lon_array.flatten()
    dist = np.sqrt((flat_lat - lat) ** 2 + (flat_lon - lon) ** 2)
    idx = dist.argmin()
    row, col = np.unravel_index(idx, lat_array.shape)
    trop_no2 = data['nitrogendioxide_tropospheric_column'][row, col]
    strat_no2 = data['nitrogendioxide_stratospheric_column'][row, col]
    total_no2 = data['nitrogendioxide_total_column'][row, col]
    return trop_no2, strat_no2, total_no2


def process_excel_and_update_data(excel_path, tropomi_folder):
    """
    读取包含tropomi_time、latitude、longitude的Excel数据集，
    根据每行的tropomi_time转换为YYYYMMDD格式，
    在指定文件夹中匹配对应日期的TROPOMI文件（每天可能有多个），
    并提取对应经纬度的NO₂浓度，写回Excel中新增列：
       tropomi_trop_no2, tropomi_strat_no, tropomi_no2
    """
    df = pd.read_excel(excel_path)
    df['tropomi_trop_no2'] = np.nan
    df['tropomi_strat_no'] = np.nan
    df['tropomi_no2'] = np.nan

    file_list = glob.glob(os.path.join(tropomi_folder, '*.nc'))
    date_to_files = {}
    for file in file_list:
        filename = os.path.basename(file)
        try:
            date_str = extract_date_from_filename(filename)
            date_to_files.setdefault(date_str, []).append(file)
        except Exception as e:
            print(f"处理文件 {filename} 时出错：{e}")

    for idx, row in df.iterrows():
        tropomi_time = row['tropomi_time']
        try:
            dt = pd.to_datetime(tropomi_time)
        except Exception as e:
            print(f"时间解析错误 {tropomi_time}：{e}")
            continue
        date_key = dt.strftime("%Y%m%d")
        if date_key in date_to_files:
            file_path = date_to_files[date_key][0]  # 默认取第一个文件
            try:
                data = read_tropomi_data(file_path)
                lat = row['latitude']
                lon = row['longitude']
                trop_no2, strat_no2, total_no2 = extract_no2_value(data, lat, lon)
                df.at[idx, 'tropomi_trop_no2'] = trop_no2
                df.at[idx, 'tropomi_strat_no'] = strat_no2
                df.at[idx, 'tropomi_no2'] = total_no2
            except Exception as e:
                print(f"处理文件 {file_path} 对应第 {idx} 行时出错：{e}")
        else:
            print(f"未找到日期为 {date_key} 的TROPOMI文件，对第 {idx} 行数据无法匹配。")

    output_path = excel_path.replace('.xlsx', '_updated.xlsx')
    df.to_excel(output_path, index=False)
    print(f"处理完成，更新后的文件已保存至 {output_path}")
    return output_path


def analyze_pair(df, x_col, y_col, ax=None):
    """
    对数据集中的一对变量进行分析：
      - 筛选出x和y均大于零且存在匹配值的行
      - 采用线性回归拟合，计算R（相关系数）
      - 绘制散点图、拟合线以及1:1等比例参考线
    返回该对数据的R值
    """
    df_valid = df[(df[x_col] > 0) & (df[y_col] > 0)].dropna(subset=[x_col, y_col])
    X = df_valid[x_col].values.reshape(-1, 1)
    y = df_valid[y_col].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    # 计算相关系数R，保留符号
    r = np.sign(model.coef_[0]) * np.sqrt(r2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X, y, alpha=0.5, label=f'Data (n={len(X)})')
    # 绘制回归拟合线
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color='red', label=f'Fit line (R={r:.3f})')
    # 绘制1:1参考线
    ax.plot(x_line, x_line, color='green', linestyle='--', label='1:1 line')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    ax.legend()
    return r


if __name__ == '__main__':
    # 修改下面路径为实际文件路径
    excel_path = r"D:\DBN\辐射值\processed_result_with_ERA5.xlsx"
    tropomi_folder = r"M:\TROPOMI_S5P\NO2\USA L2"

    # 更新Excel数据，提取TROPOMI对应NO₂浓度
    updated_excel = process_excel_and_update_data(excel_path, tropomi_folder)

    # 读取更新后的Excel数据进行拟合分析
    df = pd.read_excel(updated_excel)
    # 对列名做简单预处理，去除前后空格
    df.columns = [col.strip() for col in df.columns]

    # 计算观测对流层NO₂：通过每行 pgn_no2 - strat_no2 得到
    if 'pgn_no2' in df.columns and 'strat_no2' in df.columns:
        df['trop_no2'] = df['pgn_no2'] - df['strat_no2']
    else:
        raise KeyError("未找到 'pgn_no2' 或 'strat_no2' 列，请检查Excel文件中的列名。")

    # 分析三组数据：
    # 1. pgn_no2 vs tropomi_no2
    # 2. strat_no2 vs tropomi_strat_no
    # 3. 观测对流层NO₂（trop_no2） vs tropomi_trop_no2
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    r_pgn = analyze_pair(df, 'pgn_no2', 'tropomi_no2', ax=axs[0])
    r_strat = analyze_pair(df, 'strat_no2', 'tropomi_strat_no', ax=axs[1])
    r_trop = analyze_pair(df, 'trop_no2', 'tropomi_trop_no2', ax=axs[2])

    plt.tight_layout()
    plt.show()

    print(f'R for pgn_no2 vs tropomi_no2: {r_pgn:.3f}')
    print(f'R for strat_no2 vs tropomi_strat_no: {r_strat:.3f}')
    print(f'R for trop_no2 vs tropomi_trop_no2: {r_trop:.3f}')
