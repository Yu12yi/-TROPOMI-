import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import time
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import warnings
from typing import Dict, Tuple, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from numba import jit

warnings.filterwarnings('ignore')


class ERA5Matcher:
    def __init__(self, accum_file: str, instant_file: str):
        """初始化ERA5数据匹配器"""
        self.accum_ds = nc.Dataset(accum_file)
        self.instant_ds = nc.Dataset(instant_file)

        # 设置匹配阈值
        self.MAX_DISTANCE = 0.01  # 度
        self.MAX_TIME_DIFF = timedelta(minutes=15)  # 15分钟

        # 获取经纬度网格和时间
        self.lats = self.accum_ds['latitude'][:]
        self.lons = self.accum_ds['longitude'][:]
        self.times = pd.to_datetime(self.accum_ds['valid_time'][:], unit='s')

        # 创建空间索引
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        self.points = np.column_stack((self.lat_grid.ravel(), self.lon_grid.ravel()))
        self.tree = cKDTree(self.points)

        # 初始化缓存
        self.spatial_cache: Dict[Tuple[float, float], int] = {}
        self.temporal_cache: Dict[str, int] = {}

        # 获取所有变量列表
        self.accum_vars = [var for var in self.accum_ds.variables
                           if len(self.accum_ds[var].dimensions) == 3]
        self.instant_vars = [var for var in self.instant_ds.variables
                             if len(self.instant_ds[var].dimensions) == 3]

        # 预处理时间索引
        self.time_stamps = pd.to_datetime(self.accum_ds['valid_time'][:], unit='s')
        self.time_array = np.array([t.timestamp() for t in self.time_stamps])
        
        # 预计算网格点
        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing='ij')
        self.grid_points = np.column_stack((self.lat_grid.ravel(), self.lon_grid.ravel()))
        
        # 创建空间索引（使用更高效的KDTree）
        self.tree = cKDTree(self.grid_points)
        
        # 初始化结果缓存
        self.cache_size = 1000
        self.spatial_cache = {}
        self.temporal_cache = {}
        
        # 预加载常用数据
        self.preload_data()

    def preload_data(self):
        """预加载所有ERA5数据到内存"""
        self.accum_data = {}
        self.instant_data = {}
        
        # 需要的ERA5变量列表
        required_vars = {
            'tp': 'Total Precipitation',
            'cdir': 'Clear-Sky Direct Solar Radiation at Surface',
            'tisr': 'TOA Incident Solar Radiation',
            'bld': 'Boundary Layer Dissipation',
            'u10': '10 metre U wind component',
            'v10': '10 metre V wind component',
            'd2m': '2 metre dewpoint temperature',
            't2m': '2 metre temperature',
            'hcc': 'High cloud cover',
            'slt': 'Soil type',
            'blh': 'Boundary layer height'
        }
        
        print("\n预加载ERA5变量:")
        print("-" * 50)
        
        # 加载所有变量并记录状态
        for var_name, var_desc in required_vars.items():
            loaded = False
            
            # 检查累积变量
            if var_name in self.accum_vars:
                try:
                    self.accum_data[var_name] = self.accum_ds[var_name][:]
                    loaded = True
                    source = "accumulation"
                except Exception as e:
                    print(f"警告: 无法加载累积变量 {var_name}: {e}")
            
            # 检查瞬时变量
            if not loaded and var_name in self.instant_vars:
                try:
                    self.instant_data[var_name] = self.instant_ds[var_name][:]
                    loaded = True
                    source = "instant"
                except Exception as e:
                    print(f"警告: 无法加载瞬时变量 {var_name}: {e}")
            
            # 报告加载状态
            if loaded:
                print(f"✓ {var_name:<6} - {var_desc} (从{source}文件加载)")
            else:
                print(f"✗ {var_name:<6} - {var_desc} (未找到)")
        
        # 验证所有必需变量是否都已加载
        loaded_vars = set(self.accum_data.keys()) | set(self.instant_data.keys())
        missing_vars = set(required_vars.keys()) - loaded_vars
        
        if missing_vars:
            print("\n警告: 以下变量未能加载:")
            for var in missing_vars:
                print(f"- {var} ({required_vars[var]})")
            print("\n请检查ERA5数据文件是否包含所有需要的变量")

    @staticmethod
    @jit(nopython=True)
    def calculate_weights(distances: np.ndarray) -> np.ndarray:
        """使用Numba加速权重计算"""
        weights = 1.0 / (distances + 1e-10)
        return weights / np.sum(weights)

    def find_nearest_time_indices(self, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量查找最近的时间索引"""
        target_stamps = np.array([pd.Timestamp(t).timestamp() for t in target_times])
        indices = np.zeros(len(target_times), dtype=int)
        time_diffs = np.zeros(len(target_times))
        
        for i, t in enumerate(target_stamps):
            time_diff = np.abs(self.time_array - t)
            idx = np.argmin(time_diff)
            indices[i] = idx
            time_diffs[i] = time_diff[idx] / 3600  # 转换为小时
            
        return indices, time_diffs

    def batch_spatial_query(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量空间查询"""
        points = np.column_stack((lats, lons))
        distances, indices = self.tree.query(points, k=4)
        return distances, indices

    def interpolate_batch(self, var_data: np.ndarray, spatial_indices: np.ndarray, 
                         temporal_indices: np.ndarray, spatial_weights: np.ndarray, 
                         temporal_weights: np.ndarray) -> np.ndarray:
        """批量插值计算"""
        n_samples = len(spatial_indices)
        results = np.zeros(n_samples)
        
        for i in range(n_samples):
            s_idx = spatial_indices[i]
            t_idx = temporal_indices[i]
            s_w = spatial_weights[i]
            t_w = temporal_weights[i]
            
            values = var_data[t_idx, s_idx // len(self.lons), s_idx % len(self.lons)]
            results[i] = np.sum(values * s_w * t_w)
            
        return results

    def batch_process(self, df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """批量处理数据"""
        n_samples = len(df)
        n_batches = (n_samples + batch_size - 1) // batch_size
        results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx]
            
            # 批量获取空间和时间索引
            spatial_distances, spatial_indices = self.batch_spatial_query(
                batch_df['latitude'].values, 
                batch_df['longitude'].values
            )
            temporal_indices, temporal_diffs = self.find_nearest_time_indices(
                batch_df['tropomi_time'].values
            )
            
            # 计算权重
            spatial_weights = self.calculate_weights(spatial_distances)
            temporal_weights = self.calculate_weights(temporal_diffs.reshape(-1, 1))
            
            # 处理每个变量
            batch_results = {}
            
            # 处理累积变量
            for var in self.accum_vars:
                if var in self.accum_data:
                    values = self.interpolate_batch(
                        self.accum_data[var],
                        spatial_indices,
                        temporal_indices,
                        spatial_weights,
                        temporal_weights
                    )
                    batch_results[f'ERA5_{var}'] = values
            
            # 处理瞬时变量
            for var in self.instant_vars:
                if var in self.instant_data:
                    values = self.interpolate_batch(
                        self.instant_data[var],
                        spatial_indices,
                        temporal_indices,
                        spatial_weights,
                        temporal_weights
                    )
                    batch_results[f'ERA5_{var}'] = values
            
            results.append(pd.DataFrame(batch_results, index=batch_df.index))
            
            if (i + 1) % 10 == 0:
                print(f"Processed {end_idx}/{n_samples} samples")
        
        result_df = pd.concat(results)
        
        # 验证结果
        expected_columns = [f'ERA5_{var}' for var in self.accum_vars + self.instant_vars]
        missing_columns = set(expected_columns) - set(result_df.columns)
        if missing_columns:
            print("\n警告: 以下ERA5变量在输出结果中缺失:")
            for col in missing_columns:
                print(f"- {col}")
        
        return result_df

    def close(self):
        """关闭数据集"""
        self.accum_ds.close()
        self.instant_ds.close()


def process_dataset(excel_file: str, accum_file: str, instant_file: str, output_file: str):
    """处理数据集并添加ERA5数据"""
    print("开始处理数据...")
    start_time = time.time()

    # 读取Excel文件
    df = pd.read_excel(excel_file)
    print(f"读取了 {len(df)} 行数据")

    # 初始化ERA5匹配器
    matcher = ERA5Matcher(accum_file, instant_file)

    # 批量处理数据
    era5_df = matcher.batch_process(df)

    # 合并结果
    result_df = pd.concat([df, era5_df], axis=1)

    # 数据质量检查
    print("\n数据质量报告:")
    print("-" * 50)
    nan_count = era5_df.isna().sum()
    total_rows = len(era5_df)

    for col in era5_df.columns:
        nan_percent = (nan_count[col] / total_rows) * 100
        print(f"{col}: {nan_percent:.2f}% 缺失值")

    # 保存结果
    result_df.to_excel(output_file, index=False)
    matcher.close()

    print(f"\n处理完成！")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    print(f"结果已保存到: {output_file}")


def main():
    # 文件路径
    excel_file = r"D:\DBN\辐射值\processed_result_with_neighbor_refine.xlsx"
    accum_file = r"J:\Word\ERA5-23\data_stream-oper_stepType-accum.nc"
    instant_file = r"J:\Word\ERA5-23\data_stream-oper_stepType-instant.nc"
    output_file = r"D:\DBN\辐射值\processed_result_with_ERA5.xlsx"

    # 处理数据
    process_dataset(excel_file, accum_file, instant_file, output_file)


if __name__ == "__main__":
    main()
