import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import os
import warnings
from scipy.integrate import simpson
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import re
import datetime as dt
from matplotlib.dates import DateFormatter, MonthLocator

sns.set_style("darkgrid")
# 设置全局字体为 Times New Roman 并配置数学文本渲染
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体集，与Times New Roman兼容
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = False  # 不使用LaTeX，避免依赖外部程序
rcParams['mathtext.default'] = 'regular'  # 确保数学文本正确渲染

warnings.filterwarnings('ignore')
# 修复：seaborn 不是有效的样式名称，使用兼容的样式名
plt.style.use('seaborn-v0_8')  # 使用seaborn-v0_8替代seaborn
os.makedirs('spectral_analysis', exist_ok=True)

class SpectralAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_excel(data_path)
        
        # 从列名中提取波长，不再假设固定间隔
        spectral_cols = [col for col in self.data.columns if col.startswith('radiance_') and col.endswith('nm')]
        self.wavelengths = np.array([float(re.search(r'radiance_(.+?)nm', col).group(1)) for col in spectral_cols])
        
        # 只保留405-465nm范围内的波长
        valid_idx = (self.wavelengths >= 405) & (self.wavelengths <= 465)
        self.wavelengths = self.wavelengths[valid_idx]
        self.spectral_cols = [spectral_cols[i] for i, valid in enumerate(valid_idx) if valid]
        
        self.X = self.data[self.spectral_cols].values
        self.y = self.data['pgn_no2'].values
        
        # 确保数据中有时间列，如果没有则创建一个基于索引的时间序列
        try:
            if 'tropomi_time' in self.data.columns:
                # 尝试强制转换为datetime格式，并处理任何无效值
                self.dates = pd.to_datetime(self.data['tropomi_time'], errors='coerce')
                self.data['datetime'] = self.dates  # 存储已转换的日期时间
                print(f"使用 'tropomi_time' 列作为时间索引，共 {self.dates.notna().sum()} 个有效日期")
            elif 'pgn_time' in self.data.columns:
                self.dates = pd.to_datetime(self.data['pgn_time'], errors='coerce')
                self.data['datetime'] = self.dates
                print(f"使用 'pgn_time' 列作为时间索引，共 {self.dates.notna().sum()} 个有效日期")
            else:
                # 创建一个假设的时间序列，从2021-01-01开始，每天一个样本
                print("注意：数据集中未找到时间列，将创建一个假设的时间序列用于演示")
                start_date = dt.datetime(2021, 1, 1)
                self.dates = pd.date_range(start=start_date, periods=len(self.X), freq='D')
                self.data['datetime'] = self.dates
        except Exception as e:
            print(f"处理日期时出错: {e}")
            print("创建默认日期序列...")
            self.dates = pd.date_range(start=dt.datetime(2021, 1, 1), periods=len(self.X), freq='D')
            self.data['datetime'] = self.dates
        
        # 提取年月信息，用于分组
        self.data['year'] = pd.DatetimeIndex(self.data['datetime']).year
        self.data['month'] = pd.DatetimeIndex(self.data['datetime']).month
        self.data['year_month'] = self.data['datetime'].dt.strftime('%Y-%m')
        
        # 设置现代科研风格的图表主题
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """设置统一的现代科研风格图形样式"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'Times New Roman',  # 保持Times New Roman
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.formatter.use_mathtext': True,  # 启用数学文本渲染
            'axes.formatter.limits': (-3, 4),
            'axes.formatter.useoffset': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            'mathtext.fontset': 'stix',  # 使用STIX字体集
            'mathtext.default': 'regular'  # 确保数学文本正确渲染
        })
        
        # 自定义配色方案
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'light_blue': '#9ecae1',
            'light_orange': '#ffbb78',
            'light_green': '#98df8a',
            'light_red': '#ff9896'
        }

    def format_number(self, value):
        """根据数值大小智能调整显示格式"""
        abs_val = abs(value)
        if (abs_val >= 1000) or (abs_val < 0.001):
            return f"{value:.2e}"  # 科学计数法
        elif abs_val >= 1:
            return f"{value:.4f}"  # 4位小数
        else:
            return f"{value:.6f}"  # 6位小数

    def generate_report(self):
        """生成数据分析报告"""
        stats_summary = {
            'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()),
            'wavelength_count': len(self.wavelengths),
            'sample_count': len(self.X),
            'mean_radiance': np.mean(self.X),
            'std_radiance': np.std(self.X),
            'no2_range': (np.min(self.y), np.max(self.y)),
            'no2_mean': np.mean(self.y),
            'no2_std': np.std(self.y)
        }
        
        report = ["光谱数据分析报告", "="*40, "\n"]
        report.append(f"1. 数据基本信息：")
        report.append(f"   - 样本数量：{stats_summary['sample_count']}个")
        report.append(f"   - 波长范围：{stats_summary['wavelength_range'][0]:.2f}-{stats_summary['wavelength_range'][1]:.2f}nm")
        report.append(f"   - 波长数量：{stats_summary['wavelength_count']}个")
        report.append(f"   - NO₂浓度范围：{self.format_number(stats_summary['no2_range'][0])}-{self.format_number(stats_summary['no2_range'][1])} ppb")
        
        report.append("\n2. 辐射值统计：")
        report.append(f"   - 平均辐射值：{self.format_number(stats_summary['mean_radiance'])}")
        report.append(f"   - 辐射值标准差：{self.format_number(stats_summary['std_radiance'])}")
        var_coef = (stats_summary['std_radiance']/stats_summary['mean_radiance']*100)
        report.append(f"   - 变异系数：{self.format_number(var_coef)}%")
        
        # 计算波段相关性
        wavelength_corr = np.corrcoef(self.X.T)
        mean_corr = np.mean(np.abs(wavelength_corr - np.eye(len(wavelength_corr))))
        
        report.append("\n3. 相关性分析：")
        report.append(f"   - 波段间平均相关系数：{self.format_number(mean_corr)}")
        
        # 输出报告到文件
        with open('spectral_analysis/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return stats_summary

    def basic_statistics(self):
        """基础统计分析可视化"""
        print("\n执行基础统计分析...")
        
        # 计算每个波段的基本统计量
        stats_df = pd.DataFrame({
            'wavelength': self.wavelengths,
            'mean': np.mean(self.X, axis=0),
            'std': np.std(self.X, axis=0),
            'min': np.min(self.X, axis=0),
            'max': np.max(self.X, axis=0),
            'cv': np.std(self.X, axis=0) / np.mean(self.X, axis=0) * 100,  # 变异系数
            'skew': stats.skew(self.X, axis=0),
            'kurtosis': stats.kurtosis(self.X, axis=0)
        })
        
        # 创建更美观的可视化布局
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)
        
        # 主标题带有波长范围信息
        fig.suptitle(f'Spectral Data Statistical Analysis ({stats_df.wavelength.min():.2f}-{stats_df.wavelength.max():.2f}nm)', 
                     fontsize=18, y=0.98)
        
        # Plot 1: 平均光谱响应和标准差带
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(stats_df['wavelength'], stats_df['mean'], '-', color=self.colors['primary'], lw=2.5, label='Mean')
        ax1.fill_between(stats_df['wavelength'], 
                      stats_df['mean'] - stats_df['std'],
                      stats_df['mean'] + stats_df['std'],
                      color=self.colors['light_blue'], alpha=0.4, label='±1σ')
        
        # 标记最大和最小峰值
        max_idx = stats_df['mean'].idxmax()
        min_idx = stats_df['mean'].idxmin()
        ax1.scatter(stats_df.iloc[max_idx]['wavelength'], stats_df.iloc[max_idx]['mean'], 
                 color='red', s=80, zorder=5, marker='o', label='Maximum')
        ax1.scatter(stats_df.iloc[min_idx]['wavelength'], stats_df.iloc[min_idx]['mean'], 
                 color='green', s=80, zorder=5, marker='o', label='Minimum')
        
        ax1.set_title('Mean Spectral Response', fontsize=15)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Radiance (W/sr/m²/nm)')
        ax1.legend(loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax1.grid(True, alpha=0.3)
        
        # 打印图1的说明
        print("\n-----平均光谱响应图说明-----")
        print("此图显示了所有样本在各波长处的平均辐射值，")
        print("蓝色区域表示±1个标准差范围，反映数据分散程度。")
        print("红点标记最大峰值，绿点标记最小峰值。")
        print("这有助于识别光谱中的关键特征波长。")
        
        # Plot 2: 变异系数分布
        ax2 = fig.add_subplot(gs[0, 1])
        cv_line = ax2.plot(stats_df['wavelength'], stats_df['cv'], '-', color=self.colors['secondary'], lw=2.5)
        
        # 标记高变异区域
        high_cv_threshold = stats_df['cv'].quantile(0.75)
        high_cv_mask = stats_df['cv'] > high_cv_threshold
        ax2.scatter(stats_df.loc[high_cv_mask, 'wavelength'], 
                 stats_df.loc[high_cv_mask, 'cv'], 
                 color=self.colors['quaternary'], s=60, alpha=0.7, label=f'High Variability (>Q3)')
        
        ax2.set_title('Coefficient of Variation Distribution', fontsize=15)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.legend(loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax2.grid(True, alpha=0.3)
        
        # 打印图2的说明
        print("\n-----变异系数分布图说明-----")
        print("变异系数(CV)表示每个波长处数据的相对变异性，")
        print("计算方式为标准差除以均值再乘以100%。")
        print("红点标记的高变异区域(>Q3)表示这些波长对NO₂浓度")
        print("变化可能更敏感，有潜力作为特征波长。")
        
        # Plot 3: 分布偏度和峰度
        ax3 = fig.add_subplot(gs[1, 0])
        ax3_twin = ax3.twinx()  # 创建共享x轴的双y轴
        
        l1 = ax3.plot(stats_df['wavelength'], stats_df['skew'], '-', color=self.colors['tertiary'], lw=2.5, label='Skewness')
        l2 = ax3_twin.plot(stats_df['wavelength'], stats_df['kurtosis'], '--', color=self.colors['quaternary'], lw=2.5, label='Kurtosis')
        
        # 标记零线
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        ax3.set_title('Distribution Shape Parameters', fontsize=15)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Skewness', color=self.colors['tertiary'])
        ax3_twin.set_ylabel('Kurtosis', color=self.colors['quaternary'])
        
        # 合并两个轴的图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        
        ax3.grid(True, alpha=0.3)
        ax3_twin.grid(False)  # 避免双重网格线
        
        # 打印图3的说明
        print("\n-----分布形状参数图说明-----")
        print("偏度(Skewness)表示数据分布的不对称性，")
        print("正值表示右偏，负值表示左偏。")
        print("峰度(Kurtosis)反映分布的尖锐程度，")
        print("大于0表示比正态分布更尖锐，小于0则更平坦。")
        print("这些参数有助于理解各波长处数据的分布特性。")
        
        # Plot 4: 分位数图 (更现代的箱线图替代)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 计算分位数
        q1 = np.percentile(self.X, 25, axis=0)
        q2 = np.percentile(self.X, 50, axis=0)
        q3 = np.percentile(self.X, 75, axis=0)
        p95 = np.percentile(self.X, 95, axis=0)
        p5 = np.percentile(self.X, 5, axis=0)
        
        # 绘制填充区域
        ax4.fill_between(stats_df['wavelength'], p5, p95, alpha=0.2, color=self.colors['light_blue'], label='5-95th Percentile')
        ax4.fill_between(stats_df['wavelength'], q1, q3, alpha=0.4, color=self.colors['light_blue'], label='Interquartile Range')
        ax4.plot(stats_df['wavelength'], q2, '-', color=self.colors['primary'], lw=2.5, label='Median')
        
        ax4.set_title('Spectral Distribution Quantiles', fontsize=15)
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Radiance')
        ax4.legend(loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax4.grid(True, alpha=0.3)
        
        # 打印图4的说明
        print("\n-----光谱分布分位数图说明-----")
        print("此图显示了辐射值在各分位数下的分布情况，")
        print("深蓝色区域表示四分位范围(IQR)，")
        print("浅蓝色区域表示5%-95%分位数范围，")
        print("蓝线为中位数。区域宽度大的波长表示")
        print("该处辐射值变化较大，可能与NO₂浓度相关。")
        
        plt.savefig('spectral_analysis/basic_statistics.png', dpi=300)
        plt.close()
        
        return stats_df
    
    def correlation_analysis(self):
        """波长间相关性分析和特征波段识别 - 仅保留数据处理部分"""
        print("\n执行相关性分析 (仅数据处理，不生成图)...")
        
        # 计算波段间相关系数
        corr_matrix = np.corrcoef(self.X.T)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        # 计算每个波长与NO2的相关性
        no2_corr = np.array([pearsonr(self.X[:, i], self.y)[0] for i in range(self.X.shape[1])])
        
        # 识别最大相关性波长
        max_corr_idx = np.abs(no2_corr).argmax()
        max_corr_wavelength = self.wavelengths[max_corr_idx]
        max_corr_value = no2_corr[max_corr_idx]
        
        # 显著性阈值
        sig_threshold = 1.96 / np.sqrt(len(self.y))
        
        # 识别显著相关的波段
        significant_mask = np.abs(no2_corr) > sig_threshold
        significant_wavelengths = self.wavelengths[significant_mask]
        significant_regions = []
        
        if len(significant_wavelengths) > 0:
            current_region = [significant_wavelengths[0]]
            for i in range(1, len(significant_wavelengths)):
                if significant_wavelengths[i] - current_region[-1] <= 1.0:
                    current_region.append(significant_wavelengths[i])
                else:
                    significant_regions.append(current_region)
                    current_region = [significant_wavelengths[i]]
            significant_regions.append(current_region)
            
        # 打印关键波长信息
        print("\n-----NO₂相关性分析统计-----")
        print(f"最大相关系数: {max_corr_value:.3f}，位于波长 {max_corr_wavelength:.1f}nm")
        print(f"平均绝对相关系数: {np.mean(np.abs(no2_corr)):.3f}")
        print(f"中位数绝对相关系数: {np.median(np.abs(no2_corr)):.3f}")
        
        if significant_regions:
            print("\n关键波长带:")
            for i, region in enumerate(significant_regions):
                if len(region) > 1:
                    print(f"波段 {i+1}: {min(region):.1f}-{max(region):.1f}nm")
        
        return {'wavelength_corr': corr_matrix, 'no2_corr': no2_corr}
    
    def spectral_feature_analysis(self):
        """光谱特征提取与分析 - 仅保留数据处理部分"""
        print("\n执行光谱特征分析 (仅数据处理，不生成图)...")
        
        # 计算各种光谱特征
        features = {}
        
        # 基本统计特征
        features['Integral'] = np.trapz(self.X, self.wavelengths, axis=1)
        features['Max/Min Ratio'] = np.max(self.X, axis=1) / np.min(self.X, axis=1)
        
        # 一阶导数特征
        first_derivative = np.gradient(self.X, axis=1)
        features['Max Gradient'] = np.max(np.abs(first_derivative), axis=1)
        
        # 计算特征与NO2浓度相关性
        correlations = {}
        for name, feature in features.items():
            corr, p_value = pearsonr(feature, self.y)
            correlations[name] = {'corr': corr, 'p_value': p_value}
            
            # 打印信息
            print(f"\n-----{name}特征分析-----")
            if name == 'Integral':
                print("特征说明: 曲线下面积，代表光谱的总能量")
            elif name == 'Max/Min Ratio':
                print("特征说明: 最大值/最小值比率，表示光谱对比度")
            else:
                print("特征说明: 最大梯度值，表示光谱变化最剧烈处")
            
            print(f"相关系数(R): {correlations[name]['corr']:.3f}")
            print(f"决定系数(R²): {correlations[name]['corr']**2:.3f}")
            print(f"p值: {correlations[name]['p_value']:.4f}")
            print(f"显著性: {'Significant' if correlations[name]['p_value'] < 0.05 else 'Not significant'}")
        
        return {'features': features, 'correlations': correlations}
    
    def derivative_analysis(self):
        """光谱导数分析 - 仅保留数据处理部分"""
        print("\n执行导数分析 (仅数据处理，不生成图)...")
        
        # 计算一阶和二阶导数
        first_derivative = np.gradient(self.X, axis=1)
        second_derivative = np.gradient(first_derivative, axis=1)
        
        # 计算平均导数
        mean_first_deriv = np.mean(first_derivative, axis=0)
        mean_second_deriv = np.mean(second_derivative, axis=0)
        
        # 计算一阶导数的零点 (斜率变化点)
        zero_crossings = np.where(np.diff(np.signbit(mean_first_deriv)))[0]
        inflection_wavelengths = [self.wavelengths[i] for i in zero_crossings]
        
        # 计算二阶导数的零点 (变化率的极值点)
        zero_crossings_2nd = np.where(np.diff(np.signbit(mean_second_deriv)))[0]
        critical_wavelengths = [self.wavelengths[i] for i in zero_crossings_2nd]
        
        # 计算每个波长处的NO2相关性
        no2_corr = np.array([pearsonr(self.X[:, i], self.y)[0] for i in range(self.X.shape[1])])
        
        # 显著性阈值
        sig_threshold = 1.96 / np.sqrt(len(self.y))
        
        # 构建有用的特征波段信息
        feature_bands = []
        if inflection_wavelengths:
            for i, wl in enumerate(inflection_wavelengths):
                idx = np.argmin(np.abs(self.wavelengths - wl))
                corr_at_inflection = no2_corr[idx]
                feature_bands.append({
                    'wavelength': wl,
                    'type': 'Inflection Point',
                    'correlation': corr_at_inflection,
                    'significance': 'Significant' if abs(corr_at_inflection) > sig_threshold else 'Not Significant'
                })
        
        print("\n-----光谱导数分析结果-----")
        print(f"找到 {len(inflection_wavelengths)} 个一阶导数零点（光谱拐点）")
        print(f"找到 {len(critical_wavelengths)} 个二阶导数零点（光谱曲率变化点）")
        print("重要特征波段:")
        for band in feature_bands:
            if band['significance'] == 'Significant':
                print(f"波长: {band['wavelength']:.2f}nm, 相关性: {band['correlation']:.3f}, 类型: {band['type']}")
        
        return {
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            'inflection_points': inflection_wavelengths,
            'critical_points': critical_wavelengths,
            'feature_bands': feature_bands
        }
    
    def spectral_time_evolution(self):
        """光谱随时间的演化分析"""
        print("\n分析光谱随时间的演化...")
        
        # 设置时间作为横坐标，直接使用所有数据点而不是按月平均
        time_positions = np.arange(len(self.dates))
        
        # 为了简化标注，每个月只显示一个标签
        month_markers = {}  # {year_month: position}
        visible_ticks = []
        visible_labels = []
        
        # 获取每个年月的第一个时间点作为标注位置
        for i, date in enumerate(self.dates):
            year_month = date.strftime('%Y-%m')
            if year_month not in month_markers:
                month_markers[year_month] = i
                visible_ticks.append(i)
                visible_labels.append(year_month)
        
        # 创建年份分隔标记
        years = sorted(list(set([date.year for date in self.dates])))
        year_boundaries = []
        year_centers = []
        
        for year in years:
            year_points = [i for i, date in enumerate(self.dates) if date.year == year]
            if year_points:
                year_boundaries.append(year_points[0])
                year_centers.append(np.mean(year_points))
        
        # 创建随时间变化的现代科研风格热图
        fig = plt.figure(figsize=(15, 11))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.25)
        
        # 热图显示所有时间点的光谱
        ax1 = fig.add_subplot(gs[0])
        
        # 使用原始数据创建热图时，确保数据维度匹配
        X, Y = np.meshgrid(time_positions, self.wavelengths)
        
        # 检查数据中是否有NaN值，如果有则填充为该波长的平均值
        data_to_plot = self.X.T.copy()  # 创建副本避免修改原始数据
        if np.isnan(data_to_plot).any():
            print("警告: 数据中存在NaN值，将使用波长平均值填充")
            for i in range(data_to_plot.shape[0]):
                mask = np.isnan(data_to_plot[i, :])
                if mask.any():
                    valid_mean = np.nanmean(data_to_plot[i, :])
                    data_to_plot[i, mask] = valid_mean
        
        # 绘制热图
        pcm = ax1.pcolormesh(X, Y, data_to_plot, shading='auto', cmap='viridis')
        
        # 设置y轴为波长，标签旋转45度
        ax1.set_ylabel('Wavelength (nm)', fontsize=14, fontweight='bold')
        
        # 修改Y轴刻度格式为包含单位的格式，并旋转45度
        y_ticks = ax1.get_yticks()
        valid_ticks = [tick for tick in y_ticks if tick >= 405 and tick <= 465]
        
        # 确保至少有一些刻度标签
        if len(valid_ticks) < 2:
            # 如果自动生成的刻度不在范围内，手动设置一些刻度
            valid_ticks = np.linspace(405, 465, 5)
        
        y_tick_labels = [f"{tick:.0f}nm" for tick in valid_ticks]
        ax1.set_yticks(valid_ticks)
        ax1.set_yticklabels(y_tick_labels, rotation=45, ha='right')
        
        # 设置横坐标刻度位置和标签
        ax1.set_xticks(visible_ticks)
        ax1.set_xticklabels(visible_labels, rotation=45, ha='right')
        
        # 在年份变化处添加垂直分隔线
        for boundary in year_boundaries[1:]:  # 跳过第一年的开始
            ax1.axvline(x=boundary-0.5, color='white', linestyle='-', alpha=0.5, linewidth=1.5)
        
        # 在图形顶部添加年份标签
        if len(years) > 1:
            ax_top = ax1.twiny()
            ax_top.set_xlim(ax1.get_xlim())
            ax_top.set_xticks(year_centers)
            ax_top.set_xticklabels(years, fontsize=12, fontweight='bold')
            ax_top.tick_params(axis='x', which='major', pad=2)
        
        # 设置标题和颜色条
        ax1.set_title('Spectral Evolution Over Time', fontsize=16, pad=10, fontweight='bold')
        cbar = plt.colorbar(pcm, ax=ax1)
        cbar.set_label('Radiance (W/sr/m²/nm)', fontsize=12, fontweight='bold')
        
        # 标记变异性最大的波长
        spectral_var = np.var(self.X, axis=0)
        max_var_idx = np.argmax(spectral_var)
        max_var_wavelength = self.wavelengths[max_var_idx]
        
        # 在热图上添加水平线标记最大变异性波长
        ax1.axhline(y=max_var_wavelength, color='red', linestyle='--', alpha=0.7, 
                  linewidth=1.5, label=f'Max Variability: {max_var_wavelength:.1f}nm')
        ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc')
        
        # 下方添加NO₂浓度随时间变化的折线图
        ax2 = fig.add_subplot(gs[1])
        
        # 使用所有数据点，而不是按月平均
        ax2.plot(time_positions, self.y, '-', color=self.colors['quaternary'], 
               linewidth=1.8, alpha=0.8, label='NO₂ Concentration')
        
        # 添加数据点 - 使用渐变色表示浓度大小
        scatter = ax2.scatter(
            time_positions, 
            self.y, 
            c=self.y, 
            cmap='RdYlBu_r',
            s=50, 
            zorder=5,
            edgecolor='white', 
            linewidth=0.8,
            alpha=0.9
        )
        
        # 添加趋势线
        z = np.polyfit(time_positions, self.y, 1)
        p = np.poly1d(z)
        trend_x = np.array([time_positions[0], time_positions[-1]])
        trend_y = p(trend_x)
        trend_label = f"Trend: {z[0]:.2e} ppb/sample"
        
        ax2.plot(
            trend_x, 
            trend_y, 
            '--', 
            color='black', 
            linewidth=1.5,
            zorder=4,
            label=trend_label
        )
        
        # 设置坐标轴
        ax2.set_xlim(time_positions[0] - 0.5, time_positions[-1] + 0.5)
        ax2.set_xticks(visible_ticks)
        ax2.set_xticklabels(visible_labels, rotation=45, ha='right')
        ax2.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('NO₂ Concentration (ppb)', fontsize=14, fontweight='bold')
        
        # 在年份变化处添加垂直分隔线
        for boundary in year_boundaries[1:]:
            ax2.axvline(x=boundary-0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)
        
        # 添加年份标注
        for i, year in enumerate(years):
            if i < len(year_centers):
                ax2.text(year_centers[i], ax2.get_ylim()[1] * 0.9, 
                       str(year), ha='center', va='center', 
                       fontsize=12, fontweight='bold', alpha=0.7,
                       bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
        
        # 添加网格和图例
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
        
        # 设置标题
        ax2.set_title('NO₂ Concentration Over Time', fontsize=16, fontweight='bold', pad=10)
        
        # 添加颜色条来表示NO2浓度
        cbar2 = plt.colorbar(scatter, ax=ax2, pad=0.01)
        cbar2.set_label('NO₂ Concentration (ppb)', fontsize=12, fontweight='bold')
        
        # 整体布局优化
        plt.tight_layout()
        plt.savefig('spectral_analysis/spectral_time_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印说明文本
        print("\n-----光谱随时间演化图说明-----")
        print("上图是光谱随时间变化的热图，横轴为时间（每月标注一次），纵轴为波长(405-465nm)，")
        print("颜色代表辐射强度，越亮表示辐射值越高。红色虚线标记了随时间变化最大的波长。")
        print("此图显示所有原始数据点，未进行月平均，可以观察到更细致的时间变化模式。")
        
        print("\n下图展示了NO₂浓度的变化趋势，")
        print("红色线和数据点显示了所有测量值，颜色深浅表示浓度大小。")
        print("黑色虚线显示NO₂浓度的整体变化趋势。")
        print("通过比较两图可以分析光谱变化与NO₂浓度变化的关系，")
        print("并观察短期波动与长期趋势。")
        
        # 分析波长特征变化
        print("\n-----光谱特征变化分析-----")
        print(f"时间维度上变异性最大的波长: {max_var_wavelength:.2f}nm")
        var_coefficient = np.sqrt(spectral_var[max_var_idx])/np.nanmean(self.X[:, max_var_idx])*100
        print(f"该波长处的变异系数: {var_coefficient:.2f}%")
        
        # 返回分析结果
        return {
            'max_var_wavelength': max_var_wavelength,
            'var_coefficient': var_coefficient
        }

    def no2_concentration_analysis(self):
        """不同NO2浓度下的光谱特征分析"""
        print("\n分析不同NO2浓度下的光谱特性...")
        
        # 创建浓度分组
        n_groups = 5
        
        # 使用分位数分组更合理
        quantiles = np.linspace(0, 100, n_groups+1)
        group_thresholds = np.percentile(self.y, quantiles)
        
        # 将数据按照NO2浓度分组
        group_indices = []
        group_labels = []
        
        for i in range(n_groups):
            if i == n_groups - 1:
                # 最后一组，包含上限
                indices = np.where((self.y >= group_thresholds[i]) & 
                                  (self.y <= group_thresholds[i+1]))[0]
                label = f'{group_thresholds[i]:.2e}–{group_thresholds[i+1]:.2e}'
            else:
                indices = np.where((self.y >= group_thresholds[i]) & 
                                  (self.y < group_thresholds[i+1]))[0]
                label = f'{group_thresholds[i]:.2e}–{group_thresholds[i+1]:.2e}'
            
            group_indices.append(indices)
            group_labels.append(label)
        
        # 计算每组的平均光谱
        group_means = [np.mean(self.X[indices], axis=0) if len(indices) > 0 else np.zeros_like(self.wavelengths) 
                      for indices in group_indices]
        
        # 更现代的可视化设计
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.25)
        
        # Plot 1: 分组光谱
        ax1 = fig.add_subplot(gs[0])
        
        # 使用渐变色映射不同的NO2浓度组
        cmap = plt.cm.viridis
        colors = [cmap(i/n_groups) for i in range(n_groups)]
        
        for i, (mean_spectrum, label) in enumerate(zip(group_means, group_labels)):
            ax1.plot(self.wavelengths, mean_spectrum, '-', color=colors[i], 
                   linewidth=2.5, label=f'NO₂: {label} ppb')
        
        # 突出显示组间差异最大的波长
        group_spectra = np.array(group_means)
        max_diff_idx = np.argmax(np.max(group_spectra, axis=0) - np.min(group_spectra, axis=0))
        max_diff_wavelength = self.wavelengths[max_diff_idx]
        
        ax1.axvline(x=max_diff_wavelength, color='red', linestyle='--', alpha=0.7, 
                  label=f'Max Difference: {max_diff_wavelength:.2f}nm')
        
        ax1.set_title('Spectral Response Across NO₂ Concentration Groups', fontsize=16)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Mean Radiance')
        ax1.legend(loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax1.grid(True, alpha=0.3)
        
        # 打印说明文本
        print("\n-----不同NO₂浓度下的光谱响应图说明-----")
        print("此图显示了不同NO₂浓度组的平均光谱响应。")
        print("颜色从浅到深表示NO₂浓度从低到高。")
        print("红色虚线标记了组间差异最大的波长，")
        print("这通常是浓度敏感波长，最适合用于NO₂监测。")
        print("曲线间差异表明光谱对NO₂浓度变化的响应模式。")
        
        # Plot 2: 组间差异分析
        ax2 = fig.add_subplot(gs[1])
        
        # 计算最高和最低NO2浓度组之间的差异
        diff_spectrum = group_means[-1] - group_means[0]
        
        # 绘制差异光谱
        ax2.plot(self.wavelengths, diff_spectrum, '-', color=self.colors['quaternary'], 
               linewidth=2.5, label='Highest - Lowest Group')
        
        # 标记零线
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # 标记最大正差异和最大负差异
        max_pos_idx = np.argmax(diff_spectrum)
        max_neg_idx = np.argmin(diff_spectrum)
        
        ax2.scatter(self.wavelengths[max_pos_idx], diff_spectrum[max_pos_idx], 
                  s=100, color='red', zorder=5, marker='o', 
                  label=f'Max +Diff: {self.wavelengths[max_pos_idx]:.2f}nm')
        
        ax2.scatter(self.wavelengths[max_neg_idx], diff_spectrum[max_neg_idx], 
                  s=100, color='blue', zorder=5, marker='o', 
                  label=f'Max -Diff: {self.wavelengths[max_neg_idx]:.2f}nm')
        
        # 高亮差异显著的区域
        threshold = np.std(diff_spectrum) * 1.5
        for i in range(len(diff_spectrum)):
            if abs(diff_spectrum[i]) > threshold:
                ax2.axvspan(self.wavelengths[i] - 0.25, self.wavelengths[i] + 0.25, 
                          alpha=0.2, color='yellow')
        
        ax2.set_title('Spectral Difference: Highest vs. Lowest NO₂ Group', fontsize=16)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Radiance Difference')
        ax2.legend(loc='best', frameon=True, facecolor='white', edgecolor='#cccccc')
        ax2.grid(True, alpha=0.3)
        
        # 打印说明文本
        print("\n-----最高与最低NO₂浓度组差异图说明-----")
        print("此图显示最高与最低NO₂浓度组之间的光谱差异。")
        print("曲线上方(正值)表示高浓度组辐射值大于低浓度组，")
        print("下方(负值)则相反。红点标记最大正差异，蓝点标记")
        print("最大负差异。黄色区域突出显示差异超过1.5个标准差")
        print("的波长，这些波长对NO₂浓度变化最敏感。")
        
        # 打印关键结果分析
        print("\n-----浓度组差异分析主要发现-----")
        print(f"最大组间差异波长: {max_diff_wavelength:.2f}nm")
        print(f"最大正差异波长: {self.wavelengths[max_pos_idx]:.2f}nm")
        print(f"最大负差异波长: {self.wavelengths[max_neg_idx]:.2f}nm")
        print("NO₂浓度组基于分位数划分")
        print("黄色高亮区域表示显著差异波段")
        
        plt.tight_layout()
        plt.savefig('spectral_analysis/no2_group_analysis.png')
        plt.close()
        
        # 返回关键波长信息
        return {
            'group_means': group_means,
            'group_labels': group_labels,
            'max_diff_wavelength': max_diff_wavelength,
            'key_wavelengths': {
                'max_pos_diff': self.wavelengths[max_pos_idx],
                'max_neg_diff': self.wavelengths[max_neg_idx]
            }
        }

def main():
    analyzer = SpectralAnalyzer(r"D:\DBN\辐射值\processed_result_with_neighbor_refine.xlsx")
    
    # 执行分析并生成报告
    stats_summary = analyzer.generate_report()
    
    # 执行保留的绘图分析
    stats_df = analyzer.basic_statistics()
    no2_analysis = analyzer.no2_concentration_analysis()
    
    # 执行新添加的时间序列分析
    time_analysis = analyzer.spectral_time_evolution()
    
    # 执行仅数据处理的分析（不绘图）
    correlation = analyzer.correlation_analysis()
    derivatives = analyzer.derivative_analysis()
    features = analyzer.spectral_feature_analysis()
    
    # 打印分析完成信息
    print("\n=== 分析完成 ===")
    print(f"数据集大小: {analyzer.X.shape}")
    print(f"波长范围: {analyzer.wavelengths.min():.2f}-{analyzer.wavelengths.max():.2f}nm")
    print(f"波长数量: {len(analyzer.wavelengths)}个")
    print(f"时间范围: {min(analyzer.dates).strftime('%Y-%m-%d')} 至 {max(analyzer.dates).strftime('%Y-%m-%d')}")
    print(f"NO₂浓度范围: {analyzer.format_number(np.min(analyzer.y))}-{analyzer.format_number(np.max(analyzer.y))} ppb")
    print("\n所有分析结果已保存到 'spectral_analysis' 目录")

if __name__ == "__main__":
    main()