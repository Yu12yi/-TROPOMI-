import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties

def read_pgn_data(file_path):
    """读取单个PGN站点数据文件"""
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        print(f"Total lines in file: {len(lines)}")
        
        if len(lines) < 100:
            print(f"Warning: File contains too few lines ({len(lines)}), skipping...")
            return pd.DataFrame()
            
        metadata = {}
        data = []
        valid_data_count = 0
        invalid_data_count = 0
        
        for i, line in enumerate(lines[:50]):
            if "Location latitude [deg]:" in line:
                metadata['latitude'] = float(line.split(":")[1].strip())
            elif "Location longitude [deg]:" in line:
                metadata['longitude'] = float(line.split(":")[1].strip())
            elif line.startswith("--------------------"):
                metadata['data_start_idx'] = i + 2
                break
                
        if len(metadata) < 3:
            print("Missing required metadata")
            return pd.DataFrame()
            
        print(f"Found latitude: {metadata['latitude']}")
        print(f"Found longitude: {metadata['longitude']}")
        print(f"Data starts at line: {metadata['data_start_idx']}")
        
        valid_lines = [
            line.split() for line in lines[metadata['data_start_idx']:]
            if line.strip() and not line.startswith(("Column", "----"))
        ]
        
        for cols in valid_lines:
            try:
                if len(cols) < 53:
                    invalid_data_count += 1
                    continue
                    
                time = datetime.strptime(cols[0], "%Y%m%dT%H%M%S.%fZ")
                quality_flag = int(cols[35])
                
                if quality_flag not in [0, 10]:
                    invalid_data_count += 1
                    continue
                    
                total_no2 = float(cols[38]) if cols[38] != '--' else np.nan
                strat_no2 = float(cols[52]) if cols[52] != '--' else np.nan
                
                if np.isnan(total_no2) or np.isnan(strat_no2):
                    invalid_data_count += 1
                    continue
                    
                trop_no2 = total_no2 - strat_no2
                
                data.append({
                    'time': time,
                    'total_no2': total_no2,
                    'strat_no2': strat_no2,
                    'trop_no2': trop_no2,
                    'quality_flag': quality_flag
                })
                valid_data_count += 1
                
            except Exception:
                invalid_data_count += 1
                continue
        
        if not data:
            print("No valid data found in file!")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        print(f"Valid data points: {valid_data_count}")
        print(f"Invalid/filtered data points: {invalid_data_count}")
        print(f"DataFrame shape: {df.shape}")
        print(f"NO2 value range: {df['trop_no2'].min():.2e} to {df['trop_no2'].max():.2e}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def calculate_averages(df):
    """计算月度和年度平均值"""
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    
    monthly_avg = df.groupby(['station', 'year', 'month'])['trop_no2'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    yearly_avg = df.groupby(['station', 'year'])['trop_no2'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    return monthly_avg, yearly_avg

def calculate_advanced_statistics(df):
    """计算高级统计信息"""
    correlation_matrix = df[['trop_no2', 'strat_no2', 'total_no2']].corr()
    
    df['month'] = df['time'].dt.month
    seasonal_stats = df.groupby(['station', 'month']).agg({
        'trop_no2': ['mean', 'std', 'count'],
        'strat_no2': ['mean', 'std'],
        'total_no2': ['mean', 'std']
    }).round(6)
    
    trend_stats = {}
    for station in df['station'].unique():
        station_data = df[df['station'] == station]
        if len(station_data) > 0:
            X = (station_data['time'] - station_data['time'].min()).dt.total_seconds().values.reshape(-1, 1)
            y = station_data['trop_no2'].values
            model = LinearRegression()
            model.fit(X, y)
            trend_stats[station] = {
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }
    
    correlation_matrix.to_csv('correlation_matrix.csv')
    seasonal_stats.to_csv('seasonal_statistics.csv')
    pd.DataFrame(trend_stats).T.to_csv('trend_analysis.csv')
    
    return correlation_matrix, seasonal_stats, trend_stats

def plot_no2_comparison(df, no2_type, title_suffix):
    """绘制指定类型的NO2 VCD浓度比较图"""
    # 设置字体属性以正确处理中英文
    times_font = FontProperties(family='Times New Roman')
    simsun_font = FontProperties(family='SimSun')
    
    # 计算每个站点的平均浓度，用于排序和颜色映射
    station_means = df.groupby('station')[no2_type].mean().sort_values()
    stations_sorted = station_means.index.tolist()
    # 反转列表，使浓度从小到大对应从下往上（低浓度在底部）
    stations_sorted.reverse()
    
    # 获取最小和最大浓度的站点信息
    min_station = station_means.index[0]
    min_value = station_means.iloc[0]
    max_station = station_means.index[-1]
    max_value = station_means.iloc[-1]
    
    # 计算总体平均值
    overall_mean = df[no2_type].mean()
    
    # 创建一个新的颜色映射: 红色->橙色->淡黄->淡蓝->浅绿（颜色顺序已调整）
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.8, 0.0, 0.0),   # 红色
        (1.0, 0.7, 0.3),   # 橙色
        (1.0, 1.0, 0.6),   # 淡黄
        (0.6, 0.8, 0.9),   # 淡蓝
        (0.5, 0.9, 0.5)    # 浅绿
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)
    station_colors = [custom_cmap(i) for i in np.linspace(0, 1, len(stations_sorted))]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    
    # 使用Seaborn的barplot，并定制颜色和排序
    bars = sns.barplot(
        x=no2_type, 
        y='station', 
        data=df,
        estimator=np.mean,
        errorbar=None,
        order=stations_sorted,
        palette=dict(zip(stations_sorted, station_colors)),
        ax=ax
    )
    
    # 添加箱线图形式的统计信息
    boxplot = sns.boxplot(
        x=no2_type, 
        y='station', 
        data=df,
        order=stations_sorted,
        width=0.4,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        whiskerprops=dict(alpha=0.3),
        medianprops=dict(color='black'),
        ax=ax
    )
    
    # 为每个柱状图添加平均值标签
    for i, station in enumerate(stations_sorted):
        mean_value = station_means[station]
        # 在条形图右侧添加值标签，使用Times New Roman字体
        ax.text(
            mean_value * 1.02,  # 稍微偏右一点
            i,  # y位置对应站点索引
            f"{mean_value:.2e}",
            va='center',
            fontsize=9,
            fontproperties=times_font
        )
    
    # 设置图表标题和轴标签
    ax.set_title(f'2023年全年各站 {title_suffix} VCD 浓度比较图', fontsize=16, pad=20, fontproperties=simsun_font)
    ax.set_xlabel(f'平均 {title_suffix} VCD 浓度 (molecules/cm$^2$)', fontsize=14, fontproperties=simsun_font)
    ax.set_ylabel('站点', fontsize=14, fontproperties=simsun_font)
    
    # 格式化坐标轴标签字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(times_font)  # X轴标签使用Times New Roman
    
    for label in ax.get_yticklabels():
        label.set_fontproperties(simsun_font)  # Y轴标签使用宋体
    
    # 添加网格线以便于阅读
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 在右下角添加注释，包括总体均值、最大值和最小值，紧凑排列
    min_text = f"最小浓度站点: {min_station}\n浓度值: {min_value:.2e} molecules/cm$^2$"
    max_text = f"最大浓度站点: {max_station}\n浓度值: {max_value:.2e} molecules/cm$^2$"
    mean_text = f"总体平均浓度: {overall_mean:.2e} molecules/cm$^2$"
    
    # 调整注释位置，设为居左对齐，并减少行间距
    ax.text(0.70, 0.06, mean_text + '\n' + min_text + '\n' + max_text,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=11,
            linespacing=1.1,  # 减少行间距
            fontproperties=simsun_font,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'{title_suffix}_concentration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(df, monthly_avg, yearly_avg):
    """绘制三种NO2 VCD浓度比较图"""
    # 绘制对流层NO2浓度比较图
    plot_no2_comparison(df, 'trop_no2', '对流层NO$_2$')
    
    # 绘制平流层NO2浓度比较图
    plot_no2_comparison(df, 'strat_no2', '平流层NO$_2$')
    
    # 绘制总NO2浓度比较图
    plot_no2_comparison(df, 'total_no2', '总NO$_2$')

def process_all_stations(root_path):
    """处理所有站点数据"""
    all_data = []
    total_stations = 0
    successful_stations = 0
    failed_stations = []
    
    files_to_process = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if "rnvs" in file and file.endswith(".txt"):
                files_to_process.append(os.path.join(root, file))
    
    print(f"\nFound {len(files_to_process)} files to process")
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        total_stations += 1
        try:
            df = read_pgn_data(file_path)
            if not df.empty:
                station_name = file_path.split(os.sep)[-4]
                df['station'] = station_name
                all_data.append(df)
                successful_stations += 1
            else:
                failed_stations.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            failed_stations.append(os.path.basename(file_path))
    
    print(f"\nProcessing Summary:")
    print(f"Total stations processed: {total_stations}")
    print(f"Successful stations: {successful_stations}")
    print(f"Failed stations: {total_stations - successful_stations}")
    
    if failed_stations:
        print("\nFailed stations list:")
        for path in failed_stations:
            print(f"- {path}")
    
    if not all_data:
        raise ValueError("No valid data found in any station")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 只保留2023年的数据
    combined_data = combined_data[combined_data['time'].dt.year == 2023]
    print(f"\n已筛选2023年数据")
    
    print(f"\nFinal combined dataset:")
    print(f"Total rows: {combined_data.shape[0]}")
    print(f"Unique stations: {combined_data['station'].nunique()}")
    print(f"Date range: {combined_data['time'].min()} to {combined_data['time'].max()}")
    
    print("\nData quality check:")
    print(f"Missing values: {combined_data.isnull().sum().to_dict()}")
    print(f"NO2 value range: {combined_data['trop_no2'].min():.2e} to {combined_data['trop_no2'].max():.2e}")
    print(f"Columns: {combined_data.columns.tolist()}")
    
    return combined_data

def main():
    root_path = r"M:\Pandonia_Global_Network\USA"
    
    print("Processing all stations...")
    all_data = process_all_stations(root_path)
    
    print("\nCalculating averages...")
    monthly_avg, yearly_avg = calculate_averages(all_data)
    
    print("\nCalculating advanced statistics...")
    corr_matrix, seasonal_stats, trend_stats = calculate_advanced_statistics(all_data)
    
    print("\nPlotting results...")
    plot_results(all_data, monthly_avg, yearly_avg)
    
    print("\nDone!")

if __name__ == "__main__":
    main()