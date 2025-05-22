import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import warnings
from scipy import stats
import traceback
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import matplotlib as mpl
import shap  # 新增：导入SHAP库

# 设置中文字体支持 - 使用通用字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore')

# 创建结果目录
def create_result_dirs():
    base_dir = "D:\\DBN\\results"
    os.makedirs(base_dir, exist_ok=True)
    
    for target in ['pgn_no2', 'strat_no2', 'trop_no2']:
        path = os.path.join(base_dir, f"{target}_reflectance")
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "models"), exist_ok=True)
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        os.makedirs(os.path.join(path, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(path, "shap_analysis"), exist_ok=True)  # 新增：SHAP分析目录
    
    return base_dir

base_result_dir = create_result_dirs()

# 读取数据
df = pd.read_excel("D:\\DBN\\辐射值\\processed_result_with_ERA5.xlsx")

# 计算反射率和对流层NO2
def compute_reflectance_and_tropno2(df):
    # 查找所有的radiance和irradiance列
    radiance_cols = [col for col in df.columns if col.startswith('radiance_')]
    irradiance_cols = [col for col in df.columns if col.startswith('irradiance_')]
    
    # 提取波长值并确保一致性
    rad_wavelengths = sorted([float(col.replace('radiance_', '').replace('nm', '')) for col in radiance_cols])
    irr_wavelengths = sorted([float(col.replace('irradiance_', '').replace('nm', '')) for col in irradiance_cols])
    
    # 验证波长一致性
    if rad_wavelengths != irr_wavelengths:
        print("Warning: Wavelengths do not match between radiance and irradiance!")
        # 使用共同的波长
        wavelengths = sorted(list(set(rad_wavelengths).intersection(set(irr_wavelengths))))
    else:
        wavelengths = rad_wavelengths
    
    # 计算反射率
    for wl in wavelengths:
        rad_col = f'radiance_{wl}nm'
        irr_col = f'irradiance_{wl}nm'
        refl_col = f'reflectance_{wl}nm'
        
        if rad_col in df.columns and irr_col in df.columns:
            # 角度转弧度
            solar_zenith_rad = np.radians(df['solar_zenith'].values)
            
            # 计算反射率
            reflectance = (df[rad_col].values * math.pi) / (df[irr_col].values * np.cos(solar_zenith_rad))
            
            # 处理无效值
            reflectance[~np.isfinite(reflectance)] = np.nan
            mean_refl = np.nanmean(reflectance)
            reflectance = np.nan_to_num(reflectance, nan=mean_refl)
            
            df[refl_col] = reflectance
    
    # 计算对流层NO2
    if 'pgn_no2' in df.columns and 'strat_no2' in df.columns:
        df['trop_no2'] = df['pgn_no2'] - df['strat_no2']
    
    return df

# 应用反射率计算
df = compute_reflectance_and_tropno2(df)

# 保存处理后的数据集
df.to_excel("D:\\DBN\\辐射值\\processed_result_with_reflectance.xlsx", index=False)
print("Reflectance calculated and processed dataset saved")

# 计算重要的光谱特征和波段比率
def compute_spectral_indices(df, refl_cols):
    """
    Calculate common spectral indices and band ratios to enhance NO2 signal
    """
    wavelengths = [float(col.replace('reflectance_', '').replace('nm', '')) for col in refl_cols]
    wavelength_dict = {wl: col for wl, col in zip(wavelengths, refl_cols)}
    
    # 添加绿峰特征 (峰值在~550nm)
    green_cols = [col for wl, col in wavelength_dict.items() if 520 <= wl <= 570]
    if green_cols:
        df['green_peak'] = df[green_cols].mean(axis=1)
    
    # 添加蓝/红比 (蓝光吸收较强)
    blue_cols = [col for wl, col in wavelength_dict.items() if 420 <= wl <= 470]
    red_cols = [col for wl, col in wavelength_dict.items() if 620 <= wl <= 680]
    
    if blue_cols and red_cols:
        df['blue_red_ratio'] = df[blue_cols].mean(axis=1) / df[red_cols].mean(axis=1)
    
    # 添加NO2敏感波段指数 (405-465nm是NO2吸收最强区域)
    no2_sens_cols = [col for wl, col in wavelength_dict.items() if 405 <= wl <= 465]
    if no2_sens_cols:
        df['no2_index'] = df[no2_sens_cols].mean(axis=1)
    
    # 添加波段差值特征
    if blue_cols and red_cols:
        df['blue_minus_red'] = df[blue_cols].mean(axis=1) - df[red_cols].mean(axis=1)
    
    print(f"Spectral indices and band features calculated")
    return df

# 新增：使用降维和特征选择处理光谱数据
def process_spectral_features(X_spectral, y, n_components=20, n_features=30):
    """
    Use PCA dimensionality reduction and feature selection to reduce the dimensionality of spectral data
    
    Parameters:
        X_spectral: Spectral data matrix
        y: Target variable
        n_components: Number of PCA components
        n_features: Final number of features to select
        
    Returns:
        Processed feature matrix
    """
    print(f"Original spectral features: {X_spectral.shape[1]}")
    
    # 尝试PCA降维
    try:
        pca = PCA(n_components=min(n_components, X_spectral.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_spectral)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA features: {X_pca.shape[1]}, explained variance: {explained_var:.2%}")
        
        # 如果PCA效果不好，尝试直接特征选择
        if explained_var < 0.8 and X_spectral.shape[1] > n_features:
            print("Low PCA explained variance, trying direct feature selection...")
            selector = SelectKBest(f_regression, k=min(n_features, X_spectral.shape[1]))
            X_selected = selector.fit_transform(X_spectral, y)
            print(f"Feature selection features: {X_selected.shape[1]}")
            return X_selected, selector, "feature_selection"
        
        return X_pca, pca, "pca"
    
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}, using original features")
        return X_spectral, None, "original"

# 为不同波段和目标准备数据，增强了光谱特征处理
def prepare_data(df, target_type='pgn_no2', wavelength_range=(400.0, 700.0)):
    min_wl, max_wl = wavelength_range
    
    # 只选择反射率数据
    data_type = 'reflectance'
    
    # 选择对应波段范围的列
    spectral_columns = [col for col in df.columns if col.startswith(f'{data_type}_') and 
                       min_wl <= float(col.replace(f'{data_type}_', '').replace('nm', '')) <= max_wl]
    
    # 计算光谱指数
    df = compute_spectral_indices(df, spectral_columns)
    
    # 获取计算的光谱指数列
    spectral_indices = ['green_peak', 'blue_red_ratio', 'no2_index', 'blue_minus_red']
    spectral_indices = [col for col in spectral_indices if col in df.columns]
    
    # 合并光谱列和光谱指数
    all_spectral_columns = spectral_columns + spectral_indices
    
    X_spectral = df[all_spectral_columns].values
    
    # 其他特征
    # 地理位置
    geo_columns = ['latitude', 'longitude']
    X_geo = df[geo_columns].values
    
    # 气象数据
    era5_columns = [col for col in df.columns if col.startswith('ERA5_')]
    X_era5 = df[era5_columns].values
    
    # 观测几何
    geometry_columns = ['solar_zenith', 'solar_azimuth', 'viewing_zenith', 'viewing_azimuth']
    X_geometry = df[geometry_columns].values
    
    # 时间特征编码
    le = LabelEncoder()
    df['season_encoded'] = le.fit_transform(df['season'])
    df['day_type_encoded'] = le.fit_transform(df['day_type'])
    df['class_encoded'] = le.fit_transform(df['class'])
    time_columns = ['day_of_week', 'season_encoded', 'day_type_encoded', 'class_encoded']
    X_time = df[time_columns].values
    
    # 目标变量
    y = df[target_type].values
    
    # 应用降维/特征选择处理光谱数据
    X_spectral_processed, spectral_processor, processor_type = process_spectral_features(X_spectral, y, n_components=20)
    
    # 数据标准化
    scalers = {
        'spectral': StandardScaler(),
        'geo': StandardScaler(),
        'era5': StandardScaler(),
        'geometry': StandardScaler(),
        'time': StandardScaler(),
        'target': StandardScaler()
    }
    
    X_spectral_scaled = scalers['spectral'].fit_transform(X_spectral_processed)
    X_geo_scaled = scalers['geo'].fit_transform(X_geo)
    X_era5_scaled = scalers['era5'].fit_transform(X_era5)
    X_geometry_scaled = scalers['geometry'].fit_transform(X_geometry)
    X_time_scaled = scalers['time'].fit_transform(X_time)
    y_scaled = scalers['target'].fit_transform(y.reshape(-1, 1)).ravel()
    
    # 数据集分割 - 修改为三部分划分（训练/验证/测试）
    # 首先将数据分为两部分（80%训练相关，20%测试）
    train_val_indices, test_indices = train_test_split(range(len(y)), test_size=0.2, random_state=42)
    
    # 然后将训练相关部分分为训练集和验证集（原始数据的70%和10%）
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.125, random_state=42)  # 0.125 of 80% = 10% of total
    
    print(f"Dataset split - Training: {len(train_indices)} samples, Validation: {len(val_indices)} samples, Test: {len(test_indices)} samples")
    
    # 特征组织
    feature_groups = [
        {
            'name': f'{data_type}_all_features',
            'features': ['spectral', 'spatial', 'temporal', 'era5', 'geometry'],
            'sizes': {
                'spectral': X_spectral_scaled.shape[1],
                'spatial': 2,    # latitude, longitude
                'temporal': 4,   # day_of_week, season_encoded, day_type_encoded, class_encoded
                'era5': len(era5_columns),
                'geometry': 4    # solar/viewing angles
            }
        }
    ]
    
    # 准备特征名称列表，用于SHAP可视化
    feature_names = {
        'spectral': ['Spectral_' + str(i+1) for i in range(X_spectral_scaled.shape[1])],
        'spatial': geo_columns,
        'era5': era5_columns,
        'geometry': geometry_columns,
        'temporal': time_columns
    }
    
    data = {
        'X_spectral': X_spectral_scaled,
        'X_geo': X_geo_scaled,
        'X_era5': X_era5_scaled,
        'X_geometry': X_geometry_scaled,
        'X_time': X_time_scaled,
        'y': y_scaled,
        'train_indices': train_indices,
        'val_indices': val_indices,   # 新增：验证集索引
        'test_indices': test_indices,
        'feature_groups': feature_groups,
        'spectral_columns': spectral_columns,
        'spectral_indices': spectral_indices,
        'all_spectral_columns': all_spectral_columns,
        'scalers': scalers,
        'raw_target': y,
        'original_df': df,  # 保存原始数据帧，用于反演
        'processor_type': processor_type,
        'spectral_processor': spectral_processor,  # 保存处理器用于后续反演
        'feature_names': feature_names  # 新增：特征名称字典
    }
    
    return data

# 自定义数据集
class NO2Dataset(Dataset):
    def __init__(self, indices, feature_config, data_dict):
        self.indices = indices
        self.feature_config = feature_config
        self.data_dict = data_dict
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        features = {}
        
        if 'spectral' in self.feature_config['features']:
            features['spectral'] = torch.FloatTensor(self.data_dict['X_spectral'][i])
            
        if 'spatial' in self.feature_config['features']:
            features['spatial'] = torch.FloatTensor(self.data_dict['X_geo'][i])
            
        if 'temporal' in self.feature_config['features']:
            features['temporal'] = torch.FloatTensor(self.data_dict['X_time'][i])
            
        if 'era5' in self.feature_config['features']:
            features['era5'] = torch.FloatTensor(self.data_dict['X_era5'][i])
            
        if 'geometry' in self.feature_config['features']:
            features['geometry'] = torch.FloatTensor(self.data_dict['X_geometry'][i])
        
        return features, torch.FloatTensor([self.data_dict['y'][i]])

# 改进后的神经网络模型 - 光谱特有处理
class NO2Net(nn.Module):
    def __init__(self, config):
        super(NO2Net, self).__init__()
        
        self.feature_nets = nn.ModuleDict()
        self.config = config
        
        # 增强的光谱处理网络
        if 'spectral' in config['features']:
            spectral_size = config['sizes']['spectral']
            self.feature_nets['spectral'] = nn.Sequential(
                nn.Linear(spectral_size, 128),
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32)
            )
        
        if 'spatial' in config['features'] and config['sizes']['spatial'] > 0:
            self.feature_nets['spatial'] = nn.Sequential(
                nn.Linear(config['sizes']['spatial'], 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 16)
            )
            
        if 'temporal' in config['features'] and config['sizes']['temporal'] > 0:
            self.feature_nets['temporal'] = nn.Sequential(
                nn.Linear(config['sizes']['temporal'], 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 16)
            )
            
        if 'era5' in config['features'] and config['sizes']['era5'] > 0:
            self.feature_nets['era5'] = nn.Sequential(
                nn.Linear(config['sizes']['era5'], 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32)
            )
            
        if 'geometry' in config['features'] and config['sizes']['geometry'] > 0:
            self.feature_nets['geometry'] = nn.Sequential(
                nn.Linear(config['sizes']['geometry'], 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16)
            )
        
        # 计算融合层输入大小
        fusion_input_size = 0
        feature_sizes = {
            'spectral': 32,
            'spatial': 16,
            'temporal': 16,
            'era5': 32,
            'geometry': 16
        }
        
        for feature in config['features']:
            fusion_input_size += feature_sizes[feature]
        
        # 融合层 (增加层深度)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_dict):
        feature_outputs = []
        
        for feature in self.config['features']:
            if feature in x_dict:
                output = self.feature_nets[feature](x_dict[feature])
                feature_outputs.append(output)
        
        # 特征融合
        x = torch.cat(feature_outputs, dim=1)
        return self.fusion(x)

# 新增函数：PyTorch模型转为SHAP解释器模型
def create_shap_model(model, data_dict, feature_group):
    """
    创建一个可被SHAP使用的模型函数
    """
    def predict_fn(X):
        model.eval()
        device = next(model.parameters()).device
        batch_size = 128
        outputs = []
        
        # 获取特征类型和索引
        feature_indices = {}
        start_idx = 0
        
        for feature_type in feature_group['features']:
            if feature_type == 'spectral':
                size = feature_group['sizes']['spectral']
            elif feature_type == 'spatial':
                size = feature_group['sizes']['spatial']
            elif feature_type == 'temporal':
                size = feature_group['sizes']['temporal']
            elif feature_type == 'era5':
                size = feature_group['sizes']['era5']
            elif feature_type == 'geometry':
                size = feature_group['sizes']['geometry']
            
            feature_indices[feature_type] = (start_idx, start_idx + size)
            start_idx += size
        
        # 逐批次处理
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            
            # 将输入分离为不同特征组
            batch_features = {}
            for feature_type in feature_group['features']:
                start, end = feature_indices[feature_type]
                batch_features[feature_type] = torch.FloatTensor(batch_X[:, start:end]).to(device)
            
            with torch.no_grad():
                batch_out = model(batch_features).cpu().numpy()
            
            outputs.append(batch_out)
        
        return np.vstack(outputs)
    
    return predict_fn

# 新增函数：创建合并的输入数据用于SHAP分析
def create_shap_input_data(data_dict, indices, feature_group):
    """
    为SHAP分析创建一个合并的输入数据集
    """
    features_list = []
    feature_names = []
    
    for feature_type in feature_group['features']:
        if feature_type == 'spectral':
            features_list.append(data_dict['X_spectral'][indices])
            feature_names.extend(data_dict['feature_names']['spectral'])
        elif feature_type == 'spatial':
            features_list.append(data_dict['X_geo'][indices])
            feature_names.extend(data_dict['feature_names']['spatial'])
        elif feature_type == 'temporal':
            features_list.append(data_dict['X_time'][indices])
            feature_names.extend(data_dict['feature_names']['temporal'])
        elif feature_type == 'era5':
            features_list.append(data_dict['X_era5'][indices])
            feature_names.extend(data_dict['feature_names']['era5'])
        elif feature_type == 'geometry':
            features_list.append(data_dict['X_geometry'][indices])
            feature_names.extend(data_dict['feature_names']['geometry'])
    
    return np.hstack(features_list), feature_names

def run_shap_analysis(model, data_dict, feature_group, target_type, n_samples=100):
    print("\n运行SHAP分析中...")
    try:
        device = next(model.parameters()).device
        model.eval()
        
        # 准备数据
        sample_indices = np.random.choice(data_dict['test_indices'], size=min(n_samples, len(data_dict['test_indices'])), replace=False)
        X_combined, feature_names = create_shap_input_data(data_dict, sample_indices, feature_group)
        
        # 创建一个用于SHAP的模型函数
        shap_model = create_shap_model(model, data_dict, feature_group)
        
        # 创建一个SHAP解释器
        explainer = shap.KernelExplainer(shap_model, X_combined)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_combined)
        
        # 目录设置
        result_dir = os.path.join(base_result_dir, f"{target_type}_reflectance")
        shap_dir = os.path.join(result_dir, "shap_analysis")
        os.makedirs(shap_dir, exist_ok=True)
        
        # 美化设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        
        # 只保留特征重要性条形图，删除其他图表
        try:
            # 确保SHAP值处理正确
            if isinstance(shap_values, list):
                # 列表类型转数组
                shap_array = np.array(shap_values)
                if shap_array.ndim > 2:
                    shap_importance = np.abs(shap_array).mean(axis=(0, 1))
                else:
                    shap_importance = np.abs(shap_array).mean(0)
            else:
                shap_importance = np.abs(shap_values).mean(0)
                
            # 验证shap_importance是一维数组
            if shap_importance.ndim > 1:
                shap_importance = shap_importance.flatten()
                
            # 获取前N个重要特征
            top_n = min(15, len(feature_names))
            indices = np.argsort(shap_importance)[-top_n:]
            
            # 创建新图形
            plt.figure(figsize=(14, 12), dpi=300, facecolor='white')
            ax = plt.gca()
            
            # 准备绘图数据 - 确保所有输入都是一维数组或标量
            y_pos = np.arange(len(indices), dtype=int)
            importance_values = shap_importance[indices].astype(float)  # 确保为浮点数
            
            # 单独绘制每个条形，避免数组参数问题
            for i, (y, value) in enumerate(zip(y_pos, importance_values)):
                # 使用明确的标量值而不是数组
                single_bar = ax.barh(
                    float(y),         # y位置（标量）
                    float(value),     # 条形宽度（标量）
                    height=0.7,       # 固定高度
                    align='center'    # 对齐方式
                )
                
                # 为单个条形设置颜色
                if len(single_bar) > 0:  # 确保条形被创建
                    bar = single_bar[0]  # 获取第一个（唯一的）条形
                    color_val = 0.1 + 0.8 * (i / len(y_pos))  # 计算颜色值
                    bar.set_color(plt.cm.viridis(color_val))   # 设置颜色
                    bar.set_alpha(0.85)                       # 设置透明度
                    
                    # 添加渐变效果（如果需要）
                    x, y_rect = bar.get_xy()
                    w, h = bar.get_width(), bar.get_height()
                    
                    # 只对条形顶部应用渐变效果
                    rect_y = y_rect + 0.2 * h
                    rect_height = 0.6 * h
                    extent = [x, x + w, rect_y, rect_y + rect_height]
                    
                    # 创建渐变
                    gradient = np.linspace(0, 1, 100).reshape(1, -1)
                    gradient = np.repeat(gradient, 10, axis=0)
                    
                    ax.imshow(gradient, aspect='auto', extent=extent, 
                            alpha=0.2, origin='lower', cmap='Blues')
                    
                    # 添加数值标签
                    ax.text(float(value) + float(value)*0.03, float(y), f'{float(value):.4f}', 
                            va='center', fontweight='bold', fontsize=10)
            
            # 设置Y轴标签 - 确保标签长度适当
            y_labels = [feature_names[i] for i in indices]
            safe_labels = []
            for lbl in y_labels:
                if isinstance(lbl, str):
                    if len(lbl) > 25:
                        safe_labels.append(lbl[:22]+'...')
                    else:
                        safe_labels.append(lbl)
                else:
                    safe_labels.append(str(lbl))  # 确保是字符串
            
            plt.yticks(y_pos, safe_labels)
            
            # 美化图表
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            
            # 添加网格线增强可读性
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
            
            # 修改标题和标签 - 简化，移除R²
            ax.set_title(f'{target_type.upper()} 特征重要性排名', 
                       fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('平均|SHAP值|', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"shap_importance_bar_{feature_group['name']}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"创建特征重要性条形图时出错: {e}")
            print(traceback.format_exc())
            plt.close()
        
        print(f"SHAP分析完成。结果保存到 {shap_dir}")
        return shap_values, feature_names
        
    except Exception as e:
        print(f"SHAP分析过程中出错: {e}")
        print(traceback.format_exc())
        # 关闭所有图形
        plt.close('all')
        # 返回空值
        return None, feature_names if 'feature_names' in locals() else []

# 训练和评估函数 - 删除训练历史绘图部分
def train_and_evaluate(data_dict, feature_group, target_type, epochs=200):
    # 只处理反射率数据
    data_type = 'reflectance'
    
    # 创建数据集和数据加载器
    train_dataset = NO2Dataset(data_dict['train_indices'], feature_group, data_dict)
    val_dataset = NO2Dataset(data_dict['val_indices'], feature_group, data_dict)
    test_dataset = NO2Dataset(data_dict['test_indices'], feature_group, data_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = NO2Net(feature_group)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    # 使用AdamW，增加权重衰减以减轻过拟合
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练参数
    best_val_loss = float('inf')
    early_stopping_patience = 20
    early_stopping_counter = 0
    
    # 结果目录和文件路径
    result_dir = os.path.join(base_result_dir, f"{target_type}_{data_type}")
    model_path = os.path.join(result_dir, "models", f"model_{feature_group['name']}.pt")
    plot_path = os.path.join(result_dir, "plots", f"performance_{feature_group['name']}.png")
    metrics_path = os.path.join(result_dir, "metrics", f"metrics_{feature_group['name']}.csv")
    
    # 记录训练历史 - 但不生成图表
    history = {
        'train_loss': [],
        'val_loss': [],
        'r2': [],
        'learning_rate': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            # 将数据移至设备
            batch_features = {k: v.to(device) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段 - 使用验证集
        model.eval()
        val_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = {k: v.to(device) for k, v in batch_features.items()}
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_targets).item()
                
                # 收集预测结果
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 计算R2分数
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 反标准化预测和目标
        predictions_original = data_dict['scalers']['target'].inverse_transform(predictions)
        targets_original = data_dict['scalers']['target'].inverse_transform(targets)
        
        r2 = 1 - np.sum((targets_original - predictions_original) ** 2) / np.sum((targets_original - np.mean(targets_original)) ** 2)
        
        # 记录历史但不绘图
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['r2'].append(r2)
        history['learning_rate'].append(current_lr)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'[{data_type}-{target_type}] Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, R2: {r2:.6f}, LR: {current_lr:.8f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    
    # 最终评估 - 使用测试集
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            # 将数据移至设备
            batch_features_dev = {k: v.to(device) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features_dev)
            
            # 收集预测结果
            final_predictions.extend(outputs.cpu().numpy())
            final_targets.extend(batch_targets.cpu().numpy())
    
    # 反标准化
    final_predictions = data_dict['scalers']['target'].inverse_transform(np.array(final_predictions))
    final_targets = data_dict['scalers']['target'].inverse_transform(np.array(final_targets))
    
    # 计算最终指标
    mse = np.mean((final_predictions - final_targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((final_targets - final_predictions) ** 2) / np.sum((final_targets - np.mean(final_targets)) ** 2)
    
    # 计算相关系数
    pearson_r, p_value = stats.pearsonr(final_targets.flatten(), final_predictions.flatten())
    
    # 计算归一化RMSE
    nrmse = rmse / (final_targets.max() - final_targets.min()) if final_targets.max() != final_targets.min() else rmse
    
    # 计算偏差
    bias = np.mean(final_predictions - final_targets)
    
    # 保存指标
    pd.DataFrame({
        'MSE': [mse],
        'RMSE': [rmse],
        'NRMSE': [nrmse],
        'Bias': [bias],
        'R2': [r2],
        'Pearson_R': [pearson_r],
        'P_Value': [p_value]
    }).to_csv(metrics_path, index=False)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    # 美化后的科研风格1:1散点图 - 全面升级版
    plt.figure(figsize=(10, 10), dpi=150, facecolor='white')
    ax = plt.gca()

    # 计算数据点距离中心的距离，用于颜色映射
    center_x = np.mean(final_targets)
    center_y = np.mean(final_predictions)
    distances = np.sqrt((final_targets - center_x)**2 + (final_predictions - center_y)**2)
    max_dist = np.max(distances)
    normalized_dist = distances / max_dist

    # 创建红色到蓝色的颜色映射，中心为红色
    colors = plt.cm.coolwarm(1 - normalized_dist)

    # 绘制圆形散点，距离中心越近颜色越红
    scatter = plt.scatter(final_targets, final_predictions, 
                     c=colors, 
                     s=30,  # 点大小
                     alpha=0.7,
                     edgecolor='k',  # 黑色边缘
                     linewidth=0.3,  # 边缘线宽
                     marker='o')  # 圆形标记

    # 添加1:1参考线
    min_val = min(final_targets.min(), final_predictions.min())
    max_val = max(final_targets.max(), final_predictions.max())
    buffer = (max_val - min_val) * 0.05  # 添加5%的缓冲区
    plot_min = min_val - buffer
    plot_max = max_val + buffer

    plt.plot([plot_min, plot_max], [plot_min, plot_max], '--', 
            color='#404040', 
            label='1:1 Line', linewidth=1.5, alpha=0.7)

    # 添加回归线
    slope, intercept = np.polyfit(final_targets.flatten(), final_predictions.flatten(), 1)
    reg_y = slope * np.array([plot_min, plot_max]) + intercept
    plt.plot([plot_min, plot_max], reg_y, '-', 
            color='#1a1a1a', 
            label=f'y = {slope:.3f}x + {intercept:.2e}', 
            linewidth=2.0)

    # 添加科研风格的评估指标文本框
    text_str = (f'$R^2$ = {r2:.3f}\n'
                f'$R$ = {pearson_r:.3f}\n'
                f'RMSE = {rmse:.3e}\n'
                f'Bias = {bias:.3e}\n'
                f'N = {len(final_targets)}')

    plt.text(0.05, 0.95, text_str, 
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                    edgecolor='#a6a6a6', pad=0.5, linewidth=1.0))

    # 美化坐标轴和网格
    plt.grid(True, linestyle='--', alpha=0.3, color='grey', linewidth=0.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#262626')

    # 添加标签和标题，确保中文正确显示
    if target_type == 'pgn_no2':
        target_name = "总NO2"
    elif target_type == 'strat_no2':
        target_name = "平流层NO2"
    elif target_type == 'trop_no2':
        target_name = "对流层NO2"
    else:
        target_name = target_type
    
    plt.xlabel(f'观测 {target_name} (mol/cm$^2$)', fontsize=14, fontweight='bold')
    plt.ylabel(f'预测 {target_name} (mol/cm$^2$)', fontsize=14, fontweight='bold')
    plt.title(f'性能评估: {target_name}', fontsize=16, fontweight='bold', pad=15)

    # 图例位置调整
    plt.legend(loc='lower right', fontsize=12, frameon=True, framealpha=0.8, edgecolor='#a6a6a6')

    # 设置轴范围
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    # 保证轴刻度标签清晰可读
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 使用高质量保存
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 删除训练历史绘图部分，不再生成training_history图表
    
    print(f'\n[{data_type}-{target_type}] Final Test Results:')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2 Score: {r2:.6f}')
    print(f'Pearson R: {pearson_r:.6f}')
    print(f'Bias: {bias:.6f}')
    
    # 执行SHAP分析
    shap_values, feature_names = run_shap_analysis(model, data_dict, feature_group, target_type)
    
    # 返回评估结果
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'bias': bias,
        'model': model,
        'feature_group': feature_group,
        'data_dict': data_dict,
        'shap_values': shap_values,
        'feature_names': feature_names
    }

# 修改后的实验运行函数 - 添加全局变量存储结果
def run_experiments():
    targets = ['pgn_no2', 'strat_no2', 'trop_no2']
    
    global all_results  # 声明为全局变量，供SHAP分析访问R²值
    all_results = {}
    
    for target_type in targets:
        print(f"\n=== Training Model: reflectance → {target_type} ===")
        
        # 准备数据
        data = prepare_data(df, target_type=target_type)
        
        # 训练和评估
        result = train_and_evaluate(data, data['feature_groups'][0], target_type)
        
        all_results[target_type] = result
    
    # 比较不同目标的结果
    compare_no2_results(all_results)

# 更新后的比较不同NO2类型的结果
def compare_no2_results(all_results):
    # 提取评估指标
    r2_scores = {target: results['r2'] for target, results in all_results.items()}
    rmse_scores = {target: results['rmse'] for target, results in all_results.items()}
    bias_scores = {target: results.get('bias', 0) for target, results in all_results.items()}
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建科研风格的比较图表
    plt.figure(figsize=(15, 10), dpi=150, facecolor='white')
    
    # 设置风格
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    
    # 使用GridSpec创建布局
    gs = gridspec.GridSpec(1, 1)
    
    # 准备数据并翻译标签
    target_names = {
        'pgn_no2': '总NO2',
        'strat_no2': '平流层NO2',
        'trop_no2': '对流层NO2'
    }
    
    # 准备数据
    metrics = pd.DataFrame({
        'R²': [r2_scores[t] for t in r2_scores],
        'RMSE (×10⁻⁵)': [rmse_scores[t] * 1e5 for t in rmse_scores],
        'Bias (×10⁻⁵)': [bias_scores[t] * 1e5 for t in bias_scores]
    }, index=[target_names[t] for t in r2_scores.keys()])
    
    # 绘制条形图
    ax = plt.subplot(gs[0])
    bar_width = 0.25
    x = np.arange(len(metrics.index))
    
    # 使用更专业的配色和风格
    ax.bar(x - bar_width, metrics['R²'], bar_width, label='R²',
         color='#0C5DA5', edgecolor='black', linewidth=0.8)
    ax.bar(x, metrics['RMSE (×10⁻⁵)'], bar_width, label='RMSE (×10⁻⁵)',
         color='#FF2C00', edgecolor='black', linewidth=0.8)
    ax.bar(x + bar_width, metrics['Bias (×10⁻⁵)'], bar_width, label='Bias (×10⁻⁵)',
         color='#00B945', edgecolor='black', linewidth=0.8)
    
    # 添加数据标签
    for i, v in enumerate(metrics['R²']):
        ax.text(i - bar_width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    for i, v in enumerate(metrics['RMSE (×10⁻⁵)']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
    for i, v in enumerate(metrics['Bias (×10⁻⁵)']):
        ax.text(i + bar_width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 设置标题和标签
    ax.set_title('模型性能指标比较', fontsize=16, fontweight='bold')
    ax.set_ylabel('指标值', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.index, rotation=0, fontsize=12)
    
    # 美化图表
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 增加图例
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(base_result_dir, "no2_comparison_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 执行实验
if __name__ == "__main__":
    run_experiments()