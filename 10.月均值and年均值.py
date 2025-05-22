import os
import glob
import datetime
import numpy as np
import xarray as xr
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
import re
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import time
import calendar
from collections import defaultdict
import gc  # Added for garbage collection
import argparse  # For command-line arguments
import sys
import logging  # For better logging
import multiprocessing as mp
from functools import partial
import h5py
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class TropomiNO2Processor:
    def __init__(self, data_dir="M:/TROPOMI_S5P/NO2/USA L2", output_dir="M:/processed_results"):
        """
        Initialize the TROPOMI NO2 data processor with optimized settings.
        """
        # Set up logging
        self.setup_logging()

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.shapefile_path = "M:/American_shap/States.shp"

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Directory for monthly and yearly averages
        self.monthly_dir = os.path.join(output_dir, "monthly_averages")
        self.yearly_dir = os.path.join(output_dir, "yearly_averages")

        # Create cache directory
        self.cache_dir = os.path.join(output_dir, "cache")

        for dir_path in [self.monthly_dir, self.yearly_dir, self.cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Variables to extract - stored in efficient format
        self.variables = {
            'latitude': 'PRODUCT/latitude',
            'longitude': 'PRODUCT/longitude',
            'nitrogendioxide_tropospheric_column': 'PRODUCT/nitrogendioxide_tropospheric_column',
            'nitrogendioxide_stratospheric_column': 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_stratospheric_column',
            'nitrogendioxide_total_column': 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_total_column'
        }

        # Quality flag path for filtering valid pixels
        self.qa_flag_path = 'PRODUCT/qa_value'
        self.qa_threshold = -0.1  # Pixels with qa_value > 0.75 are good

        # Define USA bounds for interpolation grid
        self.lat_min, self.lat_max = 24, 50  # USA latitude range
        self.lon_min, self.lon_max = -125, -66  # USA longitude range
        self.grid_resolution = 0.05  # Higher resolution grid (was 1.0)

        # Interpolation and smoothing parameters
        self.interpolation_method = 'linear'  # Options: linear, cubic, nearest
        self.smoothing_sigma = 0.3  # Gaussian smoothing parameter (0 = no smoothing)

        # Set larger batch size for efficiency
        self.batch_size = 5000000  # 5 million points per batch

        # Determine number of CPU cores to use (leave one free)
        self.n_cores = max(1, mp.cpu_count() - 1)
        self.logger.info(f"Using {self.n_cores} CPU cores for parallel processing")

        # Data caching system
        self.data_cache = {}
        self.max_cache_size = 3  # Maximum number of datasets to keep in memory

        # 加载shapefile
        self.us_states = self.load_shapefile()

        # Initialize data tracking dictionaries
        self.dates_by_month = defaultdict(list)
        self.dates_by_year = defaultdict(list)

        # Record processed dates
        self.processed_dates = self.find_processed_dates()
        self.logger.info(f"Found {len(self.processed_dates)} already processed dates")

        # Store US bounds for faster filtering
        if self.us_states is not None:
            try:
                try:
                    self.us_bounds = self.us_states.union_all().bounds  # minx, miny, maxx, maxy
                except AttributeError:
                    self.us_bounds = self.us_states.unary_union.bounds
            except Exception as e:
                self.logger.error(f"Error getting US bounds: {str(e)}")
                self.us_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        else:
            self.us_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def setup_logging(self):
        """Set up logging configuration with improved formatting"""
        self.logger = logging.getLogger('TropomiProcessor')
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(ch)

            # Also create a file handler
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            fh = logging.FileHandler(
                os.path.join(log_dir, f'tropomi_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def find_processed_dates(self):
        """Find dates that have already been processed using faster directory scanning"""
        processed_dates = set()

        # Use more efficient directory scanning
        date_pattern = re.compile(r'\d{8}')
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path) and date_pattern.match(item):
                date_str = item
                npz_file = os.path.join(item_path, f"{date_str}_original_data.npz")
                if os.path.exists(npz_file):
                    processed_dates.add(date_str)

        return processed_dates

    def update_aggregation_tracking(self):
        """Update the month and year tracking with already processed dates"""
        for date_str in self.processed_dates:
            year = date_str[:4]
            month = date_str[4:6]
            self.dates_by_month[(year, month)].append(date_str)
            self.dates_by_year[year].append(date_str)

    def load_shapefile(self):
        """加载USA州界shapefile并优化内存使用"""
        try:
            # 只加载必要的几何和属性列，减少内存使用
            states = gpd.read_file(self.shapefile_path, bbox=(self.lon_min, self.lat_min, self.lon_max, self.lat_max))
            self.logger.info(f"成功加载Shapefile，共{len(states)}个州/区域")
            return states
        except Exception as e:
            self.logger.error(f"加载Shapefile失败: {str(e)}")
            return None

    def efficient_spatial_filtering(self, data):
        """使用更高效的空间过滤方法，不创建数百万个Point对象"""
        start_time = time.time()
        self.logger.info("开始空间过滤...")

        if self.us_states is None:
            self.logger.warning("无法使用Shapefile过滤，将返回原始数据")
            return data

        # 检查数据是否为空
        if len(data['latitude']) == 0:
            self.logger.warning("输入数据为空，无需过滤")
            return data

        # 使用numpy进行边界框初步过滤（非常高效）
        bbox_mask = (
                (data['longitude'] >= self.us_bounds[0]) &
                (data['longitude'] <= self.us_bounds[2]) &
                (data['latitude'] >= self.us_bounds[1]) &
                (data['latitude'] <= self.us_bounds[3])
        )

        # 应用边界框过滤
        filtered_data = {}
        for var_name in data:
            filtered_data[var_name] = data[var_name][bbox_mask]

        self.logger.info(f"边界框过滤后数据点: {len(filtered_data['latitude'])}")

        # 使用基于栅格的高效空间过滤
        result = self.grid_based_spatial_filtering(filtered_data)

        self.logger.info(f"空间过滤前数据点: {len(data['latitude'])}, 过滤后: {len(result['latitude'])}")
        self.logger.info(f"空间过滤耗时: {time.time() - start_time:.2f}秒")

        return result

    def grid_based_spatial_filtering(self, data, cell_size=0.5):
        """基于栅格的高效空间过滤方法"""
        try:
            # 尝试获取USA的几何形状
            try:
                usa_shape = self.us_states.union_all()
            except AttributeError:
                usa_shape = self.us_states.unary_union

            # 创建栅格
            lon_cells = np.arange(self.lon_min, self.lon_max + cell_size, cell_size)
            lat_cells = np.arange(self.lat_min, self.lat_max + cell_size, cell_size)

            # 快速计算每个点所在的栅格单元
            lon_idx = np.clip(np.digitize(data['longitude'], lon_cells) - 1, 0, len(lon_cells) - 2)
            lat_idx = np.clip(np.digitize(data['latitude'], lat_cells) - 1, 0, len(lat_cells) - 2)

            # 创建唯一栅格单元标识符
            grid_ids = lat_idx * len(lon_cells) + lon_idx
            unique_grid_ids = np.unique(grid_ids)

            # 检查每个栅格中心是否在USA内
            valid_grids = set()

            for grid_id in unique_grid_ids:
                lat_i = grid_id // len(lon_cells)
                lon_i = grid_id % len(lon_cells)

                if (lat_i < 0 or lat_i >= len(lat_cells) - 1 or
                        lon_i < 0 or lon_i >= len(lon_cells) - 1):
                    continue

                cell_center_lon = (lon_cells[lon_i] + lon_cells[lon_i + 1]) / 2
                cell_center_lat = (lat_cells[lat_i] + lat_cells[lat_i + 1]) / 2

                point = Point(cell_center_lon, cell_center_lat)
                if usa_shape.contains(point):
                    valid_grids.add(grid_id)

            # 找出落在有效栅格中的点
            valid_mask = np.isin(grid_ids, list(valid_grids))

            # 应用掩码
            result = {}
            for var_name in data:
                result[var_name] = data[var_name][valid_mask]

            return result

        except Exception as e:
            self.logger.error(f"栅格过滤出错: {str(e)}")
            return data

    def extract_date_from_filename(self, filename):
        """Extract the date from a TROPOMI S5P filename."""
        match = re.search(r'_(\d{8})T', filename)
        if match:
            return match.group(1)  # Returns '20230101'
        return None

    def get_files_for_date(self, date_str):
        """Get all NetCDF files for a specific date using more efficient pattern matching."""
        pattern = os.path.join(self.data_dir, f"*{date_str}*.nc")
        return glob.glob(pattern)

    def get_all_dates(self):
        """Get all available dates from the file names with optimization."""
        self.logger.info("正在获取所有可用日期...")

        # 使用字典而不是集合，避免重复的字符串对象
        dates_dict = {}

        # 更高效的文件匹配模式
        file_pattern = os.path.join(self.data_dir, "*.nc")

        # 批量处理文件列表以减少内存使用
        for batch_files in self.batch_files(glob.glob(file_pattern), 1000):
            for f in batch_files:
                date = self.extract_date_from_filename(os.path.basename(f))
                if date:
                    dates_dict[date] = 1

        return sorted(dates_dict.keys())

    def batch_files(self, file_list, batch_size):
        """将文件列表分批处理以减少内存使用"""
        for i in range(0, len(file_list), batch_size):
            yield file_list[i:i + batch_size]

    def read_data_from_file(self, file_path):
        """Read and extract data from a NetCDF file with improved error handling and qa filtering."""
        try:
            data = {}
            with nc.Dataset(file_path, 'r') as dataset:
                # First extract coordinates and qa values
                for var_name in ['latitude', 'longitude']:
                    var_path = self.variables[var_name]
                    path_parts = var_path.split('/')
                    current = dataset

                    # Navigate to the right group
                    for part in path_parts[:-1]:
                        if part in current.groups:
                            current = current.groups[part]
                        else:
                            raise KeyError(f"Group '{part}' not found")

                    # Get variable
                    var_name_in_file = path_parts[-1]
                    if var_name_in_file in current.variables:
                        var_data = current.variables[var_name_in_file][:]
                        # Handle possible time dimension
                        if var_data.ndim == 3 and var_data.shape[0] == 1:
                            data[var_name] = var_data[0].astype(np.float32)  # Use float32 for memory efficiency
                        else:
                            data[var_name] = var_data.astype(np.float32)
                    else:
                        raise KeyError(f"Variable '{var_name_in_file}' not found")

                # Extract QA values
                qa_path_parts = self.qa_flag_path.split('/')
                current = dataset
                for part in qa_path_parts[:-1]:
                    if part in current.groups:
                        current = current.groups[part]
                    else:
                        raise KeyError(f"Group '{part}' not found in QA path")

                qa_name = qa_path_parts[-1]
                if qa_name in current.variables:
                    qa_values = current.variables[qa_name][:]
                    if qa_values.ndim == 3 and qa_values.shape[0] == 1:
                        qa_values = qa_values[0]

                    # Create QA mask
                    qa_mask = qa_values >= self.qa_threshold
                else:
                    self.logger.warning(f"QA values not found, proceeding without QA filtering")
                    qa_mask = np.ones(data['latitude'].shape, dtype=bool)

                # Now extract data variables and apply QA filtering
                for var_name in ['nitrogendioxide_tropospheric_column',
                                 'nitrogendioxide_stratospheric_column',
                                 'nitrogendioxide_total_column']:
                    var_path = self.variables[var_name]
                    path_parts = var_path.split('/')
                    current = dataset

                    # Navigate to the right group
                    for part in path_parts[:-1]:
                        if part in current.groups:
                            current = current.groups[part]
                        else:
                            raise KeyError(f"Group '{part}' not found")

                    # Get variable
                    var_name_in_file = path_parts[-1]
                    if var_name_in_file in current.variables:
                        var_data = current.variables[var_name_in_file][:]
                        # Handle possible time dimension
                        if var_data.ndim == 3 and var_data.shape[0] == 1:
                            data[var_name] = var_data[0].astype(np.float32)
                        else:
                            data[var_name] = var_data.astype(np.float32)
                    else:
                        raise KeyError(f"Variable '{var_name_in_file}' not found")

                # Apply QA mask to all data
                for var_name in data:
                    data[var_name] = data[var_name][qa_mask]

                return data

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def parallel_process_files(self, files):
        """并行处理多个文件以提高速度"""
        if len(files) <= 1:
            # 如果只有一个文件，直接处理而不使用并行
            results = []
            for file in files:
                data = self.read_data_from_file(file)
                if data is not None:
                    results.append(data)
            return results

        # 使用进程池并行处理文件
        with mp.Pool(processes=min(self.n_cores, len(files))) as pool:
            results = pool.map(self.read_data_from_file, files)

        # 过滤掉None结果
        return [r for r in results if r is not None]

    def combine_daily_data(self, files):
        """Combine data from multiple files with parallel processing and improved memory efficiency."""
        if not files:
            return None

        # 并行读取文件数据
        processed_data = self.parallel_process_files(files)

        if not processed_data:
            return None

        # 初始化合并的数据结构
        combined_data = {
            'latitude': [],
            'longitude': [],
            'nitrogendioxide_tropospheric_column': [],
            'nitrogendioxide_stratospheric_column': [],
            'nitrogendioxide_total_column': []
        }

        # 合并所有处理结果
        for data in processed_data:
            for var_name in combined_data:
                # 确保数据是平坦的数组
                if data[var_name].ndim > 1:
                    flat_data = data[var_name].reshape(-1)
                else:
                    flat_data = data[var_name]

                combined_data[var_name].append(flat_data)

            # 清理循环中的临时数据
            del data

        # 清理处理结果列表
        del processed_data
        gc.collect()

        # 合并所有数组
        for var_name in combined_data:
            if combined_data[var_name]:
                combined_data[var_name] = np.concatenate(combined_data[var_name])
            else:
                combined_data[var_name] = np.array([], dtype=np.float32)

        return combined_data

    def filter_valid_data(self, data):
        """Filter out invalid data points with optimized operations."""
        if data is None:
            return None

        # 一次性创建所有无效数据的掩码
        valid_mask = np.ones(len(data['latitude']), dtype=bool)

        for var_name in ['nitrogendioxide_tropospheric_column',
                         'nitrogendioxide_stratospheric_column',
                         'nitrogendioxide_total_column']:
            # 注意：logical_and是原地操作，不创建新数组
            np.logical_and(valid_mask,
                           ~np.isnan(data[var_name]) & (data[var_name] > 5e-15),
                           out=valid_mask)

        # 应用掩码
        filtered_data = {}
        for var_name in data:
            filtered_data[var_name] = data[var_name][valid_mask]

        # 使用更高效的空间过滤
        filtered_data = self.efficient_spatial_filtering(filtered_data)

        return filtered_data

    def smooth_data_boundaries(self, data, var_name, sigma=None):
        """
        对数据应用空间平滑处理，消除卫星轨道边界

        Parameters:
        -----------
        data : dict
            包含经纬度和变量数据的字典
        var_name : str
            要平滑的变量名称
        sigma : float, optional
            高斯平滑的参数，如果为None则使用默认值

        Returns:
        --------
        dict
            包含平滑后数据的字典
        """
        if sigma is None:
            sigma = self.smoothing_sigma

        if sigma <= 0 or len(data['latitude']) < 10:
            return data  # 无需平滑

        self.logger.info(f"对{var_name}应用空间平滑处理 (sigma={sigma})...")

        # 创建规则网格用于平滑
        grid_lon = np.linspace(self.lon_min, self.lon_max,
                               int((self.lon_max - self.lon_min) / self.grid_resolution) + 1)
        grid_lat = np.linspace(self.lat_min, self.lat_max,
                               int((self.lat_max - self.lat_min) / self.grid_resolution) + 1)

        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        # 先插值到规则网格
        points = np.column_stack((data['longitude'], data['latitude']))
        values = data[var_name]

        # 使用更稳健的插值方法，比griddata更快
        # 'linear', 'nearest', 'cubic' are options
        try:
            grid_values = griddata(
                points, values, (grid_lon_mesh, grid_lat_mesh),
                method=self.interpolation_method, fill_value=np.nan
            )

            # 应用高斯平滑滤波器
            grid_values_smoothed = gaussian_filter(
                grid_values, sigma=sigma, mode='nearest'
            )

            # 将NaN值从平滑结果复制回去
            mask = np.isnan(grid_values)
            grid_values_smoothed[mask] = np.nan

            # 插值回原始点
            result = data.copy()

            # 将非NaN值转换回原始点
            non_nan = ~np.isnan(grid_values_smoothed.flatten())
            if np.any(non_nan):
                valid_lons = grid_lon_mesh.flatten()[non_nan]
                valid_lats = grid_lat_mesh.flatten()[non_nan]
                valid_values = grid_values_smoothed.flatten()[non_nan]

                # 使用与griddata相同的方法插值回原始点
                interpolated = griddata(
                    np.column_stack((valid_lons, valid_lats)),
                    valid_values,
                    np.column_stack((data['longitude'], data['latitude'])),
                    method=self.interpolation_method,
                    fill_value=np.nan
                )

                # 替换任何NaN值为原始值
                nan_mask = np.isnan(interpolated)
                interpolated[nan_mask] = data[var_name][nan_mask]

                # 确保没有负值 - 添加这行修复
                interpolated = np.maximum(interpolated, 5e-15)

                result[var_name] = interpolated

            else:
                result[var_name] = data[var_name]  # 全都是NaN，回退到原始值

            return result

        except Exception as e:
            self.logger.error(f"平滑处理失败: {str(e)}")
            return data  # 失败时返回原始数据

    def process_day(self, date_str):
        """处理单日数据，添加平滑处理减少卫星轨道拼接痕迹"""
        self.logger.info(f"处理日期数据: {date_str}")
        start_time = time.time()

        # Check if already processed
        if date_str in self.processed_dates:
            self.logger.info(f"日期 {date_str} 已处理过，跳过")
            return True

        # Create output directory for this date
        date_output_dir = os.path.join(self.output_dir, date_str)
        if not os.path.exists(date_output_dir):
            os.makedirs(date_output_dir)

        # Get all files for this date
        files = self.get_files_for_date(date_str)
        if not files:
            self.logger.warning(f"未找到日期 {date_str} 的文件")
            return False

        self.logger.info(f"日期 {date_str} 找到 {len(files)} 个文件")

        # Combine data from all files
        self.logger.info("正在组合文件数据...")
        combined_data = self.combine_daily_data(files)

        if combined_data is None or any(len(combined_data[key]) == 0 for key in combined_data):
            self.logger.warning(f"日期 {date_str} 无法获得有效数据")
            return False

        # Filter valid data and clip with shapefile
        self.logger.info("正在过滤和空间裁剪数据...")
        filtered_data = self.filter_valid_data(combined_data)

        # 清理不再需要的数据
        del combined_data
        gc.collect()

        if filtered_data is None or len(filtered_data['latitude']) == 0:
            self.logger.warning(f"日期 {date_str} 没有有效数据点")
            return False

        # 对数据应用平滑处理以减少卫星轨道拼接痕迹
        self.logger.info("应用空间平滑处理消除卫星轨道边界...")
        smoothed_data = filtered_data.copy()

        for var_name in ['nitrogendioxide_tropospheric_column',
                         'nitrogendioxide_stratospheric_column',
                         'nitrogendioxide_total_column']:
            smoothed_data = self.smooth_data_boundaries(smoothed_data, var_name)

        # Save processed data using HDF5 for better performance
        self.logger.info("正在保存处理后的数据...")
        h5_file = os.path.join(date_output_dir, f"{date_str}_original_data.h5")

        with h5py.File(h5_file, 'w') as f:
            for var_name in smoothed_data:
                f.create_dataset(var_name, data=smoothed_data[var_name], compression="gzip", compression_opts=4)

        # Also save in NPZ format for backward compatibility
        np.savez(
            os.path.join(date_output_dir, f"{date_str}_original_data.npz"),
            latitude=smoothed_data['latitude'],
            longitude=smoothed_data['longitude'],
            nitrogendioxide_tropospheric_column=smoothed_data['nitrogendioxide_tropospheric_column'],
            nitrogendioxide_stratospheric_column=smoothed_data['nitrogendioxide_stratospheric_column'],
            nitrogendioxide_total_column=smoothed_data['nitrogendioxide_total_column']
        )

        # Plot processed data with enhanced visualization
        self.logger.info("正在生成可视化图表...")
        for var_name in ['nitrogendioxide_tropospheric_column',
                         'nitrogendioxide_stratospheric_column',
                         'nitrogendioxide_total_column']:
            self.enhanced_visualization(smoothed_data, date_str, var_name, date_output_dir)

        # Add to processed dates
        self.processed_dates.add(date_str)

        # Add to tracking for monthly and yearly aggregation
        year = date_str[:4]
        month = date_str[4:6]
        self.dates_by_month[(year, month)].append(date_str)
        self.dates_by_year[year].append(date_str)

        # 清理内存
        del filtered_data, smoothed_data
        gc.collect()

        self.logger.info(f"完成日期 {date_str} 的处理，总耗时: {time.time() - start_time:.2f}秒")
        return True

    def enhanced_visualization(self, data, date_str, var_name, output_dir):
        """创建改进的可视化图表，减少卫星轨道边界的视觉影响"""
        plt.figure(figsize=(12, 9))

        # 改进的色彩设置
        cmap = plt.cm.jet
        cmap.set_bad('lightgray', 0.8)  # 设置NaN值的显示颜色

        # 确保数据没有负值 - 添加这行修复
        plot_data = np.maximum(data[var_name], 0)

        # 计算颜色范围，使用更稳健的百分位数
        vmin = np.nanpercentile(plot_data, 2)  # 使用2%而不是5%
        vmax = np.nanpercentile(plot_data, 98)  # 使用98%而不是95%

        # 首先绘制美国州界作为底图
        if self.us_states is not None:
            self.us_states.boundary.plot(ax=plt.gca(), linewidth=0.5, color='black', alpha=0.7)

        # 创建改进的散点图
        sc = plt.scatter(
            data['longitude'],
            data['latitude'],
            c=plot_data,  # 使用确保无负值的数据
            cmap=cmap,
            s=10,  # 稍微增大点的大小以减少视觉上的"空洞"
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            edgecolors='none'  # 去掉点的边框以减少视觉噪声
        )

        # 改进的颜色条
        cbar = plt.colorbar(sc, label=var_name.replace('_', ' ').title())
        cbar.ax.tick_params(labelsize=10)

        # 修复日期解析和格式化问题
        formatted_date = "Invalid Date"
        try:
            # 对于YYYYMMDD格式的日期字符串
            if len(date_str) == 8:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = date_obj.strftime('%Y-%m-%d')
            # 对于YYYYMM格式的日期字符串
            elif len(date_str) == 6:
                year_part = date_str[:4]
                month_part = date_str[4:6]
                # 确保月份是有效的（01-12）
                month_num = int(month_part)
                if 1 <= month_num <= 12:
                    month_name = calendar.month_name[month_num]
                    formatted_date = f"{month_name} {year_part}"
            # 对于YYYY格式的日期字符串
            elif len(date_str) == 4:
                formatted_date = date_str  # 年份直接显示
        except Exception as e:
            self.logger.warning(f"日期格式化失败: {date_str}, 错误: {str(e)}")

        plt.title(f"{var_name.replace('_', ' ').title()} - {formatted_date}", fontsize=14)
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)

        # USA bounds
        plt.xlim(self.lon_min, self.lon_max)
        plt.ylim(self.lat_min, self.lat_max)

        # 添加网格线以便于定位
        plt.grid(linestyle='--', alpha=0.3)

        # 添加日期和数据统计注解
        stats_text = (
            f"Points: {len(data['latitude'])}\n"
            f"Min: {np.nanmin(plot_data):.2e}\n"
            f"Max: {np.nanmax(plot_data):.2e}\n"
            f"Mean: {np.nanmean(plot_data):.2e}"
        )
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Save figure with improved quality
        out_file = os.path.join(output_dir, f"{date_str}_{var_name}_original.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 主动清理Matplotlib缓存
        plt.clf()
        gc.collect()

    def log_memory_usage(self, step_name):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            self.logger.info(f"内存使用 [{step_name}]: {mem_info.rss / 1024 / 1024:.1f} MB")
        except ImportError:
            self.logger.info(f"内存监控需要安装psutil库")

    def process_all_days(self, start_date=None):
        """
        Process all available days with improved error handling and progress reporting.
        """
        # Initialize by finding previously processed dates
        self.update_aggregation_tracking()

        # Get all available dates
        all_dates = self.get_all_dates()
        if not all_dates:
            self.logger.error("没有找到可用的数据日期")
            return

        self.logger.info(f"找到 {len(all_dates)} 个可用日期")

        # Filter dates if start_date is provided
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
            self.logger.info(f"将从日期 {start_date} 开始处理，共 {len(all_dates)} 个日期")

        # 记录总进度
        total_dates = len(all_dates)
        processed_count = 0
        success_count = 0
        start_time_all = time.time()

        # Process each date
        for date_str in all_dates:
            processed_count += 1
            try:
                success = self.process_day(date_str)
                if success:
                    success_count += 1

                # 报告进度
                elapsed = time.time() - start_time_all
                if processed_count > 0:
                    estimated_total = elapsed * total_dates / processed_count
                    remaining = estimated_total - elapsed
                    self.logger.info(
                        f"进度: {processed_count}/{total_dates} ({processed_count / total_dates * 100:.1f}%) "
                        f"成功: {success_count} "
                        f"预计剩余时间: {remaining / 60:.1f}分钟"
                    )

            except Exception as e:
                self.logger.error(f"处理日期 {date_str} 时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

            # 强制清理内存
            gc.collect()

        # Calculate monthly and yearly averages
        self.logger.info(f"完成日期处理：成功 {success_count}/{total_dates}")
        self.logger.info("开始计算月度和年度平均值...")
        self.calculate_monthly_averages()
        self.calculate_yearly_averages()
        self.logger.info("完成所有数据处理")

        # 打印总耗时
        total_time = time.time() - start_time_all
        self.logger.info(f"总处理时间: {total_time / 60:.1f}分钟")

    def load_daily_data(self, date_str, use_h5=True):
        """加载预处理数据，优先使用HDF5格式以提高性能"""
        # Check cache first
        if date_str in self.data_cache:
            return self.data_cache[date_str]

        # Try loading HDF5 file first
        if use_h5:
            h5_file = os.path.join(self.output_dir, date_str, f"{date_str}_original_data.h5")
            if os.path.exists(h5_file):
                try:
                    # 打开HDF5文件但不立即加载所有数据
                    with h5py.File(h5_file, 'r') as f:
                        # 获取数据集列表
                        var_names = list(f.keys())

                        # 初始化数据字典
                        data = {}

                        # 按需加载数据
                        for var_name in var_names:
                            data[var_name] = f[var_name][:]

                    # 管理缓存
                    self.manage_cache(date_str, data)
                    return data
                except Exception as e:
                    self.logger.error(f"加载HDF5文件 {date_str} 出错: {str(e)}")
                    # 失败时尝试加载NPZ文件

        # Fall back to NPZ if HDF5 failed or doesn't exist
        npz_file = os.path.join(self.output_dir, date_str, f"{date_str}_original_data.npz")
        if os.path.exists(npz_file):
            try:
                loaded_data = np.load(npz_file)
                data = {
                    'latitude': loaded_data['latitude'],
                    'longitude': loaded_data['longitude'],
                    'nitrogendioxide_tropospheric_column': loaded_data['nitrogendioxide_tropospheric_column'],
                    'nitrogendioxide_stratospheric_column': loaded_data['nitrogendioxide_stratospheric_column'],
                    'nitrogendioxide_total_column': loaded_data['nitrogendioxide_total_column']
                }

                # 管理缓存
                self.manage_cache(date_str, data)
                return data
            except Exception as e:
                self.logger.error(f"加载NPZ文件 {date_str} 出错: {str(e)}")
                return None
        else:
            return None

    def manage_cache(self, key, data):
        """管理内存缓存，保持缓存大小在限制范围内"""
        # 添加新数据到缓存
        self.data_cache[key] = data

        # 如果缓存超过大小限制，移除最旧的项
        if len(self.data_cache) > self.max_cache_size:
            oldest_key = next(iter(self.data_cache))
            del self.data_cache[oldest_key]
            gc.collect()  # 强制垃圾回收

    def optimized_grid_aggregation(self, data, resolution=0.25):
        """
        优化的栅格聚合方法，减少内存使用并提高速度
        """
        start_time = time.time()
        self.logger.info(f"开始优化的栅格聚合 (分辨率={resolution}°)")

        # 创建栅格
        lon_grid = np.arange(self.lon_min, self.lon_max + resolution, resolution)
        lat_grid = np.arange(self.lat_min, self.lat_max + resolution, resolution)

        # 使用NumPy的数字化函数，非常高效
        lon_idx = np.clip(np.digitize(data['longitude'], lon_grid) - 1, 0, len(lon_grid) - 2)
        lat_idx = np.clip(np.digitize(data['latitude'], lat_grid) - 1, 0, len(lat_grid) - 2)

        # 计算每个点所在的栅格单元的唯一标识符
        grid_ids = lat_idx * len(lon_grid) + lon_idx

        # 为每个变量创建聚合结果
        grid_results = {}

        for var_name in ['nitrogendioxide_tropospheric_column',
                         'nitrogendioxide_stratospheric_column',
                         'nitrogendioxide_total_column']:
            # 按栅格ID分组并计算平均值
            unique_ids, inverse_indices = np.unique(grid_ids, return_inverse=True)

            # 使用NumPy的高效向量化操作
            values = data[var_name]
            grid_values = np.zeros(len(unique_ids))
            grid_counts = np.zeros(len(unique_ids))

            # 使用NumPy的add.at函数，它对索引执行原地加法
            np.add.at(grid_values, inverse_indices, values)
            np.add.at(grid_counts, inverse_indices, 1)

            # 计算平均值
            grid_avg = np.zeros_like(grid_values)
            nonzero = grid_counts > 0
            grid_avg[nonzero] = grid_values[nonzero] / grid_counts[nonzero]

            # 存储结果
            grid_results[var_name] = (unique_ids, grid_avg)

        # 转换回经纬度坐标
        result = {
            'latitude': np.zeros(len(grid_results['nitrogendioxide_tropospheric_column'][0])),
            'longitude': np.zeros(len(grid_results['nitrogendioxide_tropospheric_column'][0])),
        }

        # 计算每个栅格单元的中心坐标
        unique_ids = grid_results['nitrogendioxide_tropospheric_column'][0]
        for i, grid_id in enumerate(unique_ids):
            lat_i = grid_id // len(lon_grid)
            lon_i = grid_id % len(lon_grid)

            result['latitude'][i] = (lat_grid[lat_i] + lat_grid[lat_i + 1]) / 2
            result['longitude'][i] = (lon_grid[lon_i] + lon_grid[lon_i + 1]) / 2

        # 添加变量数据
        for var_name in grid_results:
            result[var_name] = grid_results[var_name][1]

        self.logger.info(f"栅格聚合：从 {len(data['latitude'])} 点减少到 {len(result['latitude'])} 点")
        self.logger.info(f"聚合耗时: {time.time() - start_time:.2f}秒")

        return result

    def calculate_monthly_averages(self):
        """优化的月平均值计算方法"""
        self.logger.info("计算月平均值...")

        for (year, month), date_list in self.dates_by_month.items():
            if not date_list:
                continue

            period_str = f"{year}{month}"
            self.logger.info(f"处理月平均值: {year}-{month}")

            # Check if already calculated
            monthly_file = os.path.join(self.monthly_dir, f"{period_str}_monthly_data.h5")
            if os.path.exists(monthly_file):
                self.logger.info(f"月均值 {year}-{month} 已存在，跳过")
                continue

            # 使用分块处理和累积聚合
            grid_resolution = self.grid_resolution

            # 创建栅格累积器
            lon_grid = np.arange(self.lon_min, self.lon_max + grid_resolution, grid_resolution)
            lat_grid = np.arange(self.lat_min, self.lat_max + grid_resolution, grid_resolution)

            grid_shape = (len(lat_grid) - 1, len(lon_grid) - 1)
            grid_sum = {
                'nitrogendioxide_tropospheric_column': np.zeros(grid_shape, dtype=np.float32),
                'nitrogendioxide_stratospheric_column': np.zeros(grid_shape, dtype=np.float32),
                'nitrogendioxide_total_column': np.zeros(grid_shape, dtype=np.float32)
            }
            grid_count = np.zeros(grid_shape, dtype=np.int32)

            # 处理每日数据
            days_with_data = 0

            for date_str in sorted(date_list):  # 按日期排序
                self.logger.info(f"处理日期 {date_str} ({days_with_data + 1}/{len(date_list)})")

                daily_data = self.load_daily_data(date_str)
                if daily_data is None or len(daily_data['latitude']) == 0:
                    continue

                days_with_data += 1

                # 对每日数据进行批处理
                batch_size = self.batch_size
                for i in range(0, len(daily_data['latitude']), batch_size):
                    # 提取批次数据
                    batch_data = {
                        key: daily_data[key][i:i + batch_size]
                        for key in daily_data
                    }

                    if len(batch_data['latitude']) == 0:
                        continue

                    # 计算栅格索引
                    lon_idx = np.clip(np.digitize(batch_data['longitude'], lon_grid) - 1, 0, len(lon_grid) - 2)
                    lat_idx = np.clip(np.digitize(batch_data['latitude'], lat_grid) - 1, 0, len(lat_grid) - 2)

                    # 对每个变量进行聚合
                    for var_name in grid_sum:
                        # 使用np.add.at以避免创建大量临时数组
                        np.add.at(grid_sum[var_name], (lat_idx, lon_idx), batch_data[var_name])

                    # 更新计数
                    np.add.at(grid_count, (lat_idx, lon_idx), 1)

                    # 清理批次数据
                    del batch_data, lon_idx, lat_idx
                    gc.collect()

                # 清理每日数据
                self.data_cache.pop(date_str, None)  # 从缓存中移除
                gc.collect()

            if days_with_data == 0:
                self.logger.warning(f"月份 {year}-{month} 没有有效数据")
                continue

            # 计算平均值
            grid_avg = {}
            for var_name in grid_sum:
                grid_avg[var_name] = np.zeros_like(grid_sum[var_name])
                nonzero = grid_count > 0
                grid_avg[var_name][nonzero] = grid_sum[var_name][nonzero] / grid_count[nonzero]

            # 转换为点数据格式
            monthly_data = self.grid_to_points(grid_avg, grid_count, lon_grid, lat_grid)

            # 平滑处理，减少视觉上的拼接痕迹
            for var_name in ['nitrogendioxide_tropospheric_column',
                             'nitrogendioxide_stratospheric_column',
                             'nitrogendioxide_total_column']:
                # 使用较小的平滑参数，因为数据已经是聚合过的
                monthly_data = self.smooth_data_boundaries(monthly_data, var_name, sigma=0.3)

            # 保存为HDF5格式
            with h5py.File(monthly_file, 'w') as f:
                for var_name in monthly_data:
                    f.create_dataset(var_name, data=monthly_data[var_name], compression="gzip", compression_opts=4)

            # 也保存为NPZ格式以向后兼容
            np.savez(
                os.path.join(self.monthly_dir, f"{period_str}_monthly_data.npz"),
                latitude=monthly_data['latitude'],
                longitude=monthly_data['longitude'],
                nitrogendioxide_tropospheric_column=monthly_data['nitrogendioxide_tropospheric_column'],
                nitrogendioxide_stratospheric_column=monthly_data['nitrogendioxide_stratospheric_column'],
                nitrogendioxide_total_column=monthly_data['nitrogendioxide_total_column']
            )

            # 生成可视化
            month_name = calendar.month_name[int(month)]
            display_period = f"{month_name} {year}"
            for var_name in ['nitrogendioxide_tropospheric_column',
                             'nitrogendioxide_stratospheric_column',
                             'nitrogendioxide_total_column']:
                self.enhanced_visualization(
                    monthly_data, period_str, var_name, self.monthly_dir,
                )

            # 清理内存
            del grid_sum, grid_count, grid_avg, monthly_data
            gc.collect()

            self.logger.info(f"完成月均值 {year}-{month}，使用了 {days_with_data} 天的数据")

    def grid_to_points(self, grid_data, grid_count, lon_grid, lat_grid):
        """将栅格数据转换为点数据格式"""
        # 找出有数据的栅格单元
        nonzero_mask = grid_count > 0
        y_indices, x_indices = np.where(nonzero_mask)

        # 计算栅格中心点坐标
        lon_centers = (lon_grid[:-1] + lon_grid[1:]) / 2
        lat_centers = (lat_grid[:-1] + lat_grid[1:]) / 2

        # 创建点数据
        point_data = {
            'latitude': np.array([lat_centers[y] for y in y_indices]),
            'longitude': np.array([lon_centers[x] for x in x_indices])
        }

        # 添加变量数据
        for var_name in grid_data:
            values = np.array([grid_data[var_name][y, x] for y, x in zip(y_indices, x_indices)])
            # 确保没有负值 - 添加这行修复
            point_data[var_name] = np.maximum(values, 5e-15)

        return point_data

    def calculate_yearly_averages(self):
        """优化的年平均值计算，直接从月平均值计算而不加载每日数据"""
        self.logger.info("计算年平均值...")

        for year, date_list in self.dates_by_year.items():
            if not date_list:
                continue

            self.logger.info(f"处理年平均值: {year}")

            # 检查是否已经计算过
            yearly_file = os.path.join(self.yearly_dir, f"{year}_yearly_data.h5")
            if os.path.exists(yearly_file):
                self.logger.info(f"年均值 {year} 已存在，跳过")
                continue

            # 初始化栅格累积器
            grid_resolution = self.grid_resolution
            lon_grid = np.arange(self.lon_min, self.lon_max + grid_resolution, grid_resolution)
            lat_grid = np.arange(self.lat_min, self.lat_max + grid_resolution, grid_resolution)

            grid_shape = (len(lat_grid) - 1, len(lon_grid) - 1)
            grid_sum = {
                'nitrogendioxide_tropospheric_column': np.zeros(grid_shape, dtype=np.float32),
                'nitrogendioxide_stratospheric_column': np.zeros(grid_shape, dtype=np.float32),
                'nitrogendioxide_total_column': np.zeros(grid_shape, dtype=np.float32)
            }
            grid_count = np.zeros(grid_shape, dtype=np.int32)

            # 从月均值计算年均值
            months = sorted(set(date[:6] for date in date_list))
            months_with_data = 0

            for month_str in months:
                month = month_str[4:6]
                monthly_file_h5 = os.path.join(self.monthly_dir, f"{year}{month}_monthly_data.h5")
                monthly_file_npz = os.path.join(self.monthly_dir, f"{year}{month}_monthly_data.npz")

                # 尝试加载HDF5或NPZ格式的月均值数据
                monthly_data = None

                if os.path.exists(monthly_file_h5):
                    try:
                        with h5py.File(monthly_file_h5, 'r') as f:
                            monthly_data = {key: f[key][:] for key in f.keys()}
                    except Exception as e:
                        self.logger.error(f"加载月均值HDF5文件 {year}-{month} 出错: {str(e)}")

                if monthly_data is None and os.path.exists(monthly_file_npz):
                    try:
                        loaded = np.load(monthly_file_npz)
                        monthly_data = {
                            'latitude': loaded['latitude'],
                            'longitude': loaded['longitude'],
                            'nitrogendioxide_tropospheric_column': loaded['nitrogendioxide_tropospheric_column'],
                            'nitrogendioxide_stratospheric_column': loaded['nitrogendioxide_stratospheric_column'],
                            'nitrogendioxide_total_column': loaded['nitrogendioxide_total_column']
                        }
                    except Exception as e:
                        self.logger.error(f"加载月均值NPZ文件 {year}-{month} 出错: {str(e)}")

                if monthly_data is None or len(monthly_data['latitude']) == 0:
                    self.logger.warning(f"月均值 {year}-{month} 没有有效数据")
                    continue

                months_with_data += 1
                self.logger.info(f"加载月均值数据: {year}-{month}")

                # 计算栅格索引
                lon_idx = np.clip(np.digitize(monthly_data['longitude'], lon_grid) - 1, 0, len(lon_grid) - 2)
                lat_idx = np.clip(np.digitize(monthly_data['latitude'], lat_grid) - 1, 0, len(lat_grid) - 2)

                # 对每个变量进行聚合，使用向量化操作提高效率
                for var_name in grid_sum:
                    np.add.at(grid_sum[var_name], (lat_idx, lon_idx), monthly_data[var_name])

                # 更新计数
                np.add.at(grid_count, (lat_idx, lon_idx), 1)

                # 清理月均值数据
                del monthly_data, lon_idx, lat_idx
                gc.collect()

            if months_with_data == 0:
                self.logger.warning(f"年份 {year} 没有有效的月均值数据")
                continue

            # 计算年平均值
            grid_avg = {}
            for var_name in grid_sum:
                grid_avg[var_name] = np.zeros_like(grid_sum[var_name])
                nonzero = grid_count > 0
                grid_avg[var_name][nonzero] = grid_sum[var_name][nonzero] / grid_count[nonzero]

            # 转换为点数据格式
            yearly_data = self.grid_to_points(grid_avg, grid_count, lon_grid, lat_grid)

            # 平滑处理减少视觉拼接痕迹
            for var_name in ['nitrogendioxide_tropospheric_column',
                             'nitrogendioxide_stratospheric_column',
                             'nitrogendioxide_total_column']:
                yearly_data = self.smooth_data_boundaries(yearly_data, var_name, sigma=0.3)

            # 保存为HDF5格式
            with h5py.File(yearly_file, 'w') as f:
                for var_name in yearly_data:
                    f.create_dataset(var_name, data=yearly_data[var_name], compression="gzip", compression_opts=4)

            # 也保存为NPZ格式以向后兼容
            np.savez(
                os.path.join(self.yearly_dir, f"{year}_yearly_data.npz"),
                latitude=yearly_data['latitude'],
                longitude=yearly_data['longitude'],
                nitrogendioxide_tropospheric_column=yearly_data['nitrogendioxide_tropospheric_column'],
                nitrogendioxide_stratospheric_column=yearly_data['nitrogendioxide_stratospheric_column'],
                nitrogendioxide_total_column=yearly_data['nitrogendioxide_total_column']
            )

            # 生成可视化
            for var_name in ['nitrogendioxide_tropospheric_column',
                             'nitrogendioxide_stratospheric_column',
                             'nitrogendioxide_total_column']:
                self.enhanced_visualization(
                    yearly_data, year, var_name, self.yearly_dir
                )

            # 清理内存
            del grid_sum, grid_count, grid_avg, yearly_data
            gc.collect()

            self.logger.info(f"完成年均值 {year}，使用了 {months_with_data} 个月的数据")

# Command line interface
def main():
    parser = argparse.ArgumentParser(description='优化的TROPOMI NO2数据处理器')
    parser.add_argument('--data_dir', default="M:/TROPOMI_S5P/NO2/USA L2", help='TROPOMI数据文件目录')
    parser.add_argument('--output_dir', default="M:/processed_results", help='处理结果保存目录')
    parser.add_argument('--start_date', help='开始处理的日期 (格式: YYYYMMDD)')
    parser.add_argument('--resolution', type=float, default=0.05, help='网格分辨率 (度)')
    parser.add_argument('--smoothing', type=float, default=0.3, help='平滑参数 (0表示不平滑)')

    args = parser.parse_args()

    processor = TropomiNO2Processor(data_dir=args.data_dir, output_dir=args.output_dir)
    processor.grid_resolution = args.resolution
    processor.smoothing_sigma = args.smoothing
    processor.process_all_days(start_date=args.start_date)

# Run the processor
if __name__ == "__main__":
    main()