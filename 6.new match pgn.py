import logging
import os
import chardet
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from functools import lru_cache
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PGNStation:
    """
    PGN 站点类，用于存储站点信息和匹配方法。
    """
    def __init__(self, file_path):
        """
        初始化 PGNStation 对象，从文件路径中提取站点信息，并读取静态站点数据。
        """
        self.file_path = file_path
        self.station_name, self.station_number = self._extract_station_info(file_path)
        self.latitude, self.longitude = self._read_station_location(file_path)
        self.lock = threading.Lock() # 为缓存读取添加锁

    def _extract_station_info(self, file_path):
        """
        从文件路径中提取站点名称和编号。
        """
        parts = file_path.split(os.sep)
        station_name = parts[-4]  # 倒数第四部分为站点名称
        station_info = parts[-3]  # 倒数第三部分包含站点编号信息
        station_number = ''.join(filter(str.isdigit, station_info))  # 提取编号中的数字
        station_number = int((int(station_number) - 1) / 10.0)  # 站点编号处理
        return station_name, station_number

    def _read_station_location(self, file_path):
        """
        读取 PGN 站点的经纬度信息，只读取一次。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Location latitude' in line:
                        latitude = float(line.split(':')[1].strip())
                    elif 'Location longitude' in line:
                        longitude = float(line.split(':')[1].strip())
                    if 'longitude' in locals() and 'latitude' in locals():
                        return latitude, longitude
        except UnicodeDecodeError:
            encoding = detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    if 'Location latitude' in line:
                        latitude = float(line.split(':')[1].strip())
                    elif 'Location longitude' in line:
                        longitude = float(line.split(':')[1].strip())
                    if 'longitude' in locals() and 'latitude' in locals():
                        return latitude, longitude
        except Exception as e:
            print(f"Error reading location info from {file_path}: {e}")
            return None, None # 返回 None 以指示读取失败

    @lru_cache(maxsize=128)
    def read_time_series_data(self, file_path):
        """
        使用缓存读取 PGN 站点的时间序列数据。
        """
        return self._read_pgn_data(file_path)

    def _read_pgn_data(self, file_path):
        """
        读取 PGN 数据文件的核心函数，已缓存。
        """
        times = []
        no2_values = []
        strat_no2_values = []
        total_uncertainty_values = []
        strat_uncertainty_values = []
        damf_values = []  # 新增 DAMF 值列表
        quality_flags = []

        try:
            try:
                data = pd.read_csv(file_path, skiprows=lambda x: x < 103, delimiter='\s+', encoding='utf-8')
            except UnicodeDecodeError:
                encoding = detect_encoding(file_path)
                data = pd.read_csv(file_path, skiprows=lambda x: x < 103, delimiter='\s+', encoding=encoding)

            if data.empty:
                print(f"Warning: No data found in {file_path}")
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            for _, row in data.iterrows():
                try:
                    time_str = row.iloc[0]  # UTC 列
                    no2_str = row.iloc[38]  # NO2 列
                    strat_no2_str = row.iloc[52] # 气候二氧化氮平流层柱量
                    total_uncertainty_str = row.iloc[42] # 总垂直柱量不确定性
                    strat_uncertainty_str = row.iloc[53] # 平流层柱量不确定性
                    damf_str = row.iloc[49]  # DAMF 列
                    quality_flag = row.iloc[35] # 质量标记列

                    # 转换时间
                    time_obj = datetime.strptime(time_str, "%Y%m%dT%H%M%S.%fZ")

                    # 转换数值，处理缺失值
                    if no2_str != '--' and not pd.isna(no2_str) and quality_flag == 0 or quality_flag == 10:
                        times.append(time_obj)
                        no2_values.append(float(no2_str))
                        strat_no2_values.append(float(strat_no2_str) if strat_no2_str != '--' else np.nan)
                        total_uncertainty_values.append(
                            float(total_uncertainty_str) if total_uncertainty_str != '--' else np.nan)
                        strat_uncertainty_values.append(
                            float(strat_uncertainty_str) if strat_uncertainty_str != '--' else np.nan)
                        damf_values.append(float(damf_str) if damf_str != '--' else np.nan)
                        quality_flags.append(quality_flag)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error processing row in {file_path}: {str(e)}")
                    continue

            return (
                np.array(times),
                np.array(no2_values),
                np.array(strat_no2_values),
                np.array(total_uncertainty_values),
                np.array(strat_uncertainty_values),
                np.array(damf_values)  # 返回 DAMF 数组
            )

        except Exception as e:
            print(f"Error reading PGN data file {file_path}: {str(e)}")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def match_tropomi_daily(self, tropomi_data):
        """
        匹配单日 TROPOMI 数据。
        """
        try:
            p_lat, p_lon = self.latitude, self.longitude
            if p_lat is None or p_lon is None:
                logging.warning(f"PGN Station {self.station_name} - {self.station_number} has invalid location, skipping.")
                return None

            t_lat = tropomi_data['lat']
            t_lon = tropomi_data['lon']
            t_times = tropomi_data['times']
            t_radiance = tropomi_data['radiance']
            t_irradiance = tropomi_data['irradiance']  # 读取辐照度数据
            solar_zenith = tropomi_data['solar_zenith']

            nearest_idx = find_nearest_point(p_lat, p_lon, t_lat, t_lon)

            # 检查太阳天顶角
            if solar_zenith[nearest_idx[0], nearest_idx[1]] >= 70:
                logging.info(f"太阳天顶角 {solar_zenith[nearest_idx[0], nearest_idx[1]]:.2f}° 超过阈值70°，跳过")
                return None

            center_distance = calculate_distance(
                p_lat, p_lon,
                t_lat[nearest_idx[0], nearest_idx[1]],
                t_lon[nearest_idx[0], nearest_idx[1]]
            )

            # 修改距离限制为7.5km
            d_max = 7.5
            if center_distance > d_max:
                logging.info(f"警告：距离{center_distance:.2f} km超过 {self.station_name}-{self.station_number} 最大距离{d_max}km")
                return None

            if len(t_times.shape) == 2:
                tropomi_time = t_times[nearest_idx[0], nearest_idx[1]]
                if isinstance(tropomi_time, str):
                    tropomi_time = convert_time(tropomi_time) # 确保 convert_time 函数可用
            else:
                tropomi_time = t_times[nearest_idx[0]]
                if isinstance(tropomi_time, str):
                    tropomi_time = convert_time(tropomi_time)

            # 读取 PGN 站点时间序列数据
            pgn_times, pgn_no2, pgn_strat_no2, pgn_total_uncertainty, pgn_strat_uncertainty, pgn_damf = self.read_time_series_data(self.file_path)
            if len(pgn_times) == 0:
                logging.warning(f"No valid PGN data timeseries found for station {self.station_name}-{self.station_number}")
                return None

            # 使用900s作为时间配准范围
            pgn_weighted_no2, min_time_diff = optimize_temporal_match(
                tropomi_time, pgn_times, pgn_no2, max_time_diff=900
            )
            strat_no2_weighted, _ = optimize_temporal_match(
                tropomi_time, pgn_times, pgn_strat_no2, max_time_diff=900
            )

            if pgn_weighted_no2 is None or np.isnan(pgn_weighted_no2):
                logging.warning(f"对于 {self.station_name}-{self.station_number} 没有找到有效的时间匹配。")
                return None

            # 检查NO2总量是否大于等于平流层NO2
            if pgn_weighted_no2 <= strat_no2_weighted:
                logging.warning(f"NO2总量 {pgn_weighted_no2} 小于平流层NO2 {strat_no2_weighted}，数据无效")
                return None

            # 获取其他加权值
            total_uncertainty_weighted, _ = optimize_temporal_match(
                tropomi_time, pgn_times, pgn_total_uncertainty, max_time_diff=900
            )
            strat_uncertainty_weighted, _ = optimize_temporal_match(
                tropomi_time, pgn_times, pgn_strat_uncertainty, max_time_diff=900
            )
            damf_weighted, _ = optimize_temporal_match(
                tropomi_time, pgn_times, pgn_damf, max_time_diff=900
            )

            # 使用相同的索引方式获取辐射值和辐照度
            tropomi_radiance = tropomi_data['radiance'][nearest_idx[0], nearest_idx[1], :]
            tropomi_irradiance = tropomi_data['irradiance'][:] if len(tropomi_data['irradiance'].shape) == 1 else tropomi_data['irradiance'][nearest_idx[0], nearest_idx[1], :]
            convolved_radiance = tropomi_radiance

            # 找到最接近的时间点的索引
            closest_time_idx = np.argmin(np.abs((pgn_times - tropomi_time).astype('timedelta64[s]').astype(int)))
            matched_pgn_time = pgn_times[closest_time_idx]

            result = {
                "station_number": self.station_number,
                "station_name": self.station_name,
                "latitude": p_lat,
                "longitude": p_lon,
                "distance_km": center_distance,
                "pgn_no2": pgn_weighted_no2,
                "strat_no2": strat_no2_weighted,
                "total_uncertainty": total_uncertainty_weighted,
                "strat_uncertainty": strat_uncertainty_weighted,
                "time_offset_seconds": min_time_diff,
                "tropomi_time": tropomi_time,
                "radiance": convolved_radiance.tolist(),
                "irradiance": tropomi_irradiance.tolist(),  # 添加辐照度数据
                "wavelengths": tropomi_data['wavelengths'].tolist(),
                "solar_zenith": tropomi_data['solar_zenith'][nearest_idx[0], nearest_idx[1]],
                "solar_azimuth": tropomi_data['solar_azimuth'][nearest_idx[0], nearest_idx[1]],
                "viewing_zenith": tropomi_data['viewing_zenith'][nearest_idx[0], nearest_idx[1]],
                "viewing_azimuth": tropomi_data['viewing_azimuth'][nearest_idx[0], nearest_idx[1]],
                "pgn_time": matched_pgn_time,  # 使用最接近的时间点
                "damf": damf_weighted,  # 添加 DAMF 值
            }

            # 添加质量控制检查
            if not quality_control(result):
                return None

            return result

        except Exception as e:
            logging.error(f"Error in match_tropomi_daily for station {self.station_name}-{self.station_number}: {e}")
            traceback.print_exc()
            return None

def get_pgn_stations_info(pgn_root_path):
    """
    获取所有 PGN 站点的信息，并创建 PGNStation 对象列表。
    """
    pgn_stations = []
    for root, _, files in os.walk(pgn_root_path):
        for file in files:
            if "rnvs" in file and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                station = PGNStation(file_path)
                if station.latitude is not None and station.longitude is not None: # 确保经纬度读取成功
                    pgn_stations.append(station)
                else:
                    logging.warning(f"Skipping station due to invalid location info: {file_path}")
    return pgn_stations

def read_tropomi_netcdf(file_path):
    """读取 TROPOMI 数据的关键变量，分别保存辐射值和辐照度"""
    try:
        with Dataset(file_path, 'r') as nc:
            radiance = nc.variables['radiance'][:]     # 辐射值
            irradiance = nc.variables['irradiance'][:] # 辐照度
            solar_zenith = nc.variables['solar_zenith_angle'][:]
            solar_azimuth = nc.variables['solar_azimuth_angle'][:]
            viewing_zenith = nc.variables['viewing_zenith_angle'][:]
            viewing_azimuth = nc.variables['viewing_azimuth_angle'][:]
            lat = nc.variables['latitude'][:]
            lon = nc.variables['longitude'][:]
            time_utc = nc.variables['time_utc'][:]
            wavelengths = nc.variables['wavelength'][:]

            # 不再计算反射率，保持原始数据
            return (lat, lon, time_utc, radiance, irradiance, solar_zenith, solar_azimuth,
                    viewing_zenith, viewing_azimuth, wavelengths)
    except Exception as e:
        logging.error(f"Error reading TROPOMI NetCDF file {file_path}: {e}")
        return None * 10  # 更新为10个返回值

def read_tropomi_data_daily(folder_path):
    """
    读取单日 TROPOMI 数据。
    """
    data_by_date = {}
    nc_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nc')])

    for nc_file in nc_files:
        file_path = os.path.join(folder_path, nc_file)
        try:
            tropomi_data_tuple = read_tropomi_netcdf(file_path)
            if tropomi_data_tuple[0] is None: # 检查读取是否失败
                continue # 如果读取失败，则跳过此文件
            lat, lon, time_utc, radiance, irradiance, solar_zenith, solar_azimuth, viewing_zenith, viewing_azimuth, wavelengths = tropomi_data_tuple

            data_by_date[nc_file] = {
                'lat': lat,
                'lon': lon,
                'times': time_utc,
                'radiance': radiance,
                'irradiance': irradiance,  # 新增辐照度数据
                'wavelengths': wavelengths,
                'solar_zenith': solar_zenith,
                'solar_azimuth': solar_azimuth,
                'viewing_zenith': viewing_zenith,
                'viewing_azimuth': viewing_azimuth,
                'nc_file_name': nc_file
            }
            logging.info(f"Successfully processed TROPOMI file: {nc_file}")
        except Exception as e:
            logging.error(f"Error processing TROPOMI file {file_path}: {e}")

    return data_by_date

def process_tropomi_folder_daily(tropomi_folder_path, pgn_stations, output_base_path):
    """
    处理单个 TROPOMI 数据文件夹（日期）。
    """
    data_by_nc_file = read_tropomi_data_daily(tropomi_folder_path)
    if not data_by_nc_file:
        logging.warning(f"No valid TROPOMI data found in folder: {tropomi_folder_path}")
        return

    for nc_file, tropomi_data in data_by_nc_file.items():
        start_time = os.path.basename(tropomi_folder_path).split('_')[0] # 从文件夹名称获取日期
        output_filename = f"match_result_{start_time}_{nc_file.replace('.nc', '')}.csv" # 修改输出文件名包含 nc 文件名
        output_path = os.path.join(output_base_path, start_time, output_filename) # 将结果保存在按日期组织的子文件夹中

        os.makedirs(os.path.dirname(output_path), exist_ok=True) # 确保输出目录存在
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor: # 限制线程数量，避免资源竞争
            future_to_station = {executor.submit(station.match_tropomi_daily, tropomi_data): station for station in pgn_stations}
            for future in as_completed(future_to_station):
                station = future_to_station[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"任务执行异常 for station {station.station_name}-{station.station_number}: {e}")

        if results:
            try:
                data_dict = prepare_data_dict_for_csv(results)
                df = pd.DataFrame(data_dict)
                df.to_csv(output_path, index=False)
                logging.info(f"Results saved to {output_path}")
                print(f"Results saved to: {output_path}") # 及时打印输出路径
            except Exception as e:
                logging.error(f"Error saving results to CSV {output_path}: {e}")
        else:
            logging.warning(f"No valid match results for TROPOMI file {nc_file} in {tropomi_folder_path}")


def prepare_data_dict_for_csv(results):
    """
    准备用于保存到 CSV 文件的字典数据结构。
    """
    if not results:
        return {}

    data_dict = {
        'station_number': [],
        'station_name': [],
        'latitude': [],
        'longitude': [],
        'tropomi_time': [],
        'pgn_time': [],
        'time_offset_seconds': [],
        'distance_km': [],
        'pgn_no2': [],
        'strat_no2': [],
        'total_uncertainty': [],
        'strat_uncertainty': [],
        'damf': [],  # 添加 DAMF 字段
        'solar_zenith': [],
        'solar_azimuth': [],
        'viewing_zenith': [],
        'viewing_azimuth': []
    }

    num_wavelengths = len(results[0]['radiance'])
    for i in range(num_wavelengths):
        data_dict[f'radiance_wavelength_{i}'] = []
        data_dict[f'irradiance_wavelength_{i}'] = []  # 新增辐照度波长字段

    for r in results:
        for key in data_dict:
            if key.startswith('radiance_wavelength_'):
                idx = int(key.split('_')[-1])
                data_dict[key].append(r['radiance'][idx])
            elif key.startswith('irradiance_wavelength_'):
                idx = int(key.split('_')[-1])
                data_dict[key].append(r['irradiance'][idx])
            else:
                data_dict[key].append(r[key])
    return data_dict

# 工具函数 (保持不变)
def calculate_distance(lat1, lon1, lat2, lon2):
    """计算两个经纬度点之间的球面距离（单位：公里）"""
    R = 6371  # 地球半径，单位：公里
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    a = np.clip(a, 0, 1) # 避免数值不稳定
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def convert_time(t):
    """转换时间字符串为datetime对象"""
    try:
        if not t or str(t).strip() == '':
            return None  # 返回None代表无效时间

        t_str = str(t).strip()
        formats = ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']

        for fmt in formats:
            try:
                return datetime.strptime(t_str, fmt)
            except ValueError:
                continue

        print(f"Warning: Cannot parse time {t_str}")
        return None
    except Exception as e:
        print(f"Error converting time {t}: {str(e)}")
        return None

def optimize_temporal_match(target_time, pgn_times, pgn_no2, max_time_diff=1800):
    """使用算术平均的时间匹配"""
    if target_time is None or len(pgn_times) == 0:
        return None, float('inf')

    time_diffs = np.abs((np.array(pgn_times) - target_time).astype('timedelta64[s]').astype(int))
    valid_mask = time_diffs <= max_time_diff
    valid_no2 = np.array(pgn_no2)[valid_mask]

    if valid_no2.size == 0:
        return None, float('inf')

    matched_no2 = np.mean(valid_no2)
    min_time_diff = np.min(time_diffs[valid_mask])

    return matched_no2, min_time_diff

def find_nearest_point(p_lat, p_lon, t_lat, t_lon):
    """优化的最邻近点查找函数"""
    lat_range = 0.1  # 0.1度范围
    lon_range = 0.1 / np.cos(np.radians(p_lat))  # 根据纬度调整经度范围

    lat_mask = (t_lat >= (p_lat - lat_range)) & (t_lat <= (p_lat + lat_range))
    lon_mask = (t_lon >= (p_lon - lon_range)) & (t_lon <= (p_lon + lon_range))
    combined_mask = lat_mask & lon_mask

    if not np.any(combined_mask):
        combined_diff = np.abs(t_lat - p_lat) + np.abs(t_lon - p_lon)
        return np.unravel_index(np.argmin(combined_diff), t_lat.shape)

    candidate_indices = np.where(combined_mask)
    candidate_lats = t_lat[candidate_indices]
    candidate_lons = t_lon[candidate_indices]

    distances = calculate_distance(p_lat, p_lon, candidate_lats, candidate_lons)
    min_distance_idx = np.argmin(distances)
    final_indices = (candidate_indices[0][min_distance_idx], candidate_indices[1][min_distance_idx])

    return final_indices

def detect_encoding(file_path):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            return result['encoding'] if result['encoding'] else 'utf-8'
    except Exception as e:
        print(f"Warning: Error detecting encoding for {file_path}: {str(e)}")
        return 'utf-8'  # 默认返回UTF-8编码

def quality_control(result):
    """
    数据质量控制函数
    """
    if result is None:
        return False

    # 检查NO2总量和平流层NO2的关系
    if result["pgn_no2"] <= result["strat_no2"]:
        logging.warning(f"Invalid NO2 values: total={result['pgn_no2']}, strat={result['strat_no2']}")
        return False

    # 检查不确定度
    if result["total_uncertainty"] <= 0 or result["strat_uncertainty"] <= 0:
        logging.warning("Invalid uncertainty values")
        return False

    # 检查太阳天顶角
    if result["solar_zenith"] >= 70:
        logging.warning(f"Solar zenith angle too large: {result['solar_zenith']}")
        return False

    # 检查距离
    if result["distance_km"] > 7.5:
        logging.warning(f"Distance too large: {result['distance_km']} km")
        return False

    return True

def read_tropomi_data_batch(folder_path, nc_files_batch):
    """
    读取一批TROPOMI数据文件（而非整天数据）。
    """
    data_by_nc_file = {}

    for nc_file in nc_files_batch:
        file_path = os.path.join(folder_path, nc_file)
        try:
            tropomi_data_tuple = read_tropomi_netcdf(file_path)
            if tropomi_data_tuple[0] is None:  # 检查读取是否失败
                continue  # 如果读取失败，则跳过此文件

            lat, lon, time_utc, radiance, irradiance, solar_zenith, solar_azimuth, viewing_zenith, viewing_azimuth, wavelengths = tropomi_data_tuple

            data_by_nc_file[nc_file] = {
                'lat': lat,
                'lon': lon,
                'times': time_utc,
                'radiance': radiance,
                'irradiance': irradiance,  # 新增辐照度数据
                'wavelengths': wavelengths,
                'solar_zenith': solar_zenith,
                'solar_azimuth': solar_azimuth,
                'viewing_zenith': viewing_zenith,
                'viewing_azimuth': viewing_azimuth,
                'nc_file_name': nc_file
            }
            logging.info(f"Successfully processed TROPOMI file: {nc_file}")
        except Exception as e:
            logging.error(f"Error processing TROPOMI file {file_path}: {e}")

    return data_by_nc_file

def process_tropomi_folder_daily(tropomi_folder_path, pgn_stations, output_base_path, batch_size=1):
    """
    处理单个TROPOMI数据文件夹（日期），采用批处理方式。
    每次仅处理batch_size个文件，处理完后释放内存。
    """
    # 获取文件夹中所有NC文件
    nc_files = sorted([f for f in os.listdir(tropomi_folder_path) if f.endswith('.nc')])

    if not nc_files:
        logging.warning(f"No NC files found in folder: {tropomi_folder_path}")
        return

    # 按批次处理文件
    for i in range(0, len(nc_files), batch_size):
        batch_nc_files = nc_files[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(nc_files) + batch_size - 1)//batch_size}: {batch_nc_files}")
        print(f"Processing batch {i//batch_size + 1}/{(len(nc_files) + batch_size - 1)//batch_size}: {batch_nc_files}")

        # 读取当前批次的数据
        data_by_nc_file = read_tropomi_data_batch(tropomi_folder_path, batch_nc_files)

        if not data_by_nc_file:
            logging.warning(f"No valid TROPOMI data in batch {batch_nc_files}")
            continue

        # 处理每个NC文件
        for nc_file, tropomi_data in data_by_nc_file.items():
            start_time = os.path.basename(tropomi_folder_path).split('_')[0]  # 从文件夹名称获取日期
            output_filename = f"match_result_{start_time}_{nc_file.replace('.nc', '')}.csv"  # 修改输出文件名包含nc文件名
            output_path = os.path.join(output_base_path, start_time, output_filename)  # 将结果保存在按日期组织的子文件夹中

            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保输出目录存在
            results = []

            with ThreadPoolExecutor(max_workers=1) as executor:  # 限制线程数量，避免资源竞争
                future_to_station = {executor.submit(station.match_tropomi_daily, tropomi_data): station for station in pgn_stations}
                for future in as_completed(future_to_station):
                    station = future_to_station[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"任务执行异常 for station {station.station_name}-{station.station_number}: {e}")

            if results:
                try:
                    data_dict = prepare_data_dict_for_csv(results)
                    df = pd.DataFrame(data_dict)
                    df.to_csv(output_path, index=False)
                    logging.info(f"Results saved to {output_path}")
                    print(f"Results saved to: {output_path}")  # 及时打印输出路径
                except Exception as e:
                    logging.error(f"Error saving results to CSV {output_path}: {e}")
            else:
                logging.warning(f"No valid match results for TROPOMI file {nc_file} in {tropomi_folder_path}")

        # 清理当前批次数据，释放内存
        del data_by_nc_file
        import gc
        gc.collect()  # 强制垃圾回收

        logging.info(f"Completed processing batch {i//batch_size + 1}, memory freed")

if __name__ == "__main__":
    try:
        tropomi_base_dir = r"M:\TROPOMI_S5P\NO2\USA L1B"
        pgn_root_path = r"M:\Pandonia_Global_Network\USA"
        output_base_path = r"M:\MATCH_RESULT"

        if not os.path.exists(output_base_path):
            os.makedirs(output_base_path)

        logging.info("读取 PGN 站点信息...")
        print("读取 PGN 站点信息...")
        pgn_stations = get_pgn_stations_info(pgn_root_path)
        logging.info(f"共找到 {len(pgn_stations)} 个 PGN 站点.")
        print(f"共找到 {len(pgn_stations)} 个 PGN 站点.")

        folders = sorted([f for f in os.listdir(tropomi_base_dir)
                           if os.path.isdir(os.path.join(tropomi_base_dir, f))])

        processed_folders = set()

        for folder in folders:
            if folder in processed_folders:
                continue

            tropomi_folder_path = os.path.join(tropomi_base_dir, folder)
            logging.info(f"Processing TROPOMI folder: {tropomi_folder_path}")
            print(f"\nProcessing TROPOMI folder: {tropomi_folder_path}")

            process_tropomi_folder_daily(tropomi_folder_path, pgn_stations, output_base_path, batch_size=1)

            processed_folders.add(folder)

            # 每个文件夹处理完毕后进行垃圾回收
            import gc
            gc.collect()

    except Exception as e:
        print(f"程序异常详细信息:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        logging.error(f"Program error: {str(e)}", exc_info=True)