# === 导入部分 ===
import numpy as np
import os
from netCDF4 import Dataset
import h5py
from datetime import datetime, timedelta
import zipfile
import shutil
import traceback
from scipy.signal import savgol_filter
import uuid
import tempfile

# === 1. 数据路径和配置 ===
l1b_dir = r"M:\TROPOMI_S5P\temp"

def create_temp_dir(base_path, prefix):
    """创建带有唯一标识的临时目录"""
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(base_path, f"{prefix}_{unique_id}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    return temp_dir

def find_data_folder():
    """查找包含 BD4 和 UVN 文件的文件夹（允许不同时间文件夹中的同一天数据匹配）"""
    print("\n=== 开始查找数据文件夹 ===")
    print(f"基础路径: {l1b_dir}")
    print(f"当前时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    result_dir = os.path.join(l1b_dir, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果文件夹: {result_dir}")

    if not os.path.exists(l1b_dir):
        print(f"错误: 基础路径不存在: {l1b_dir}")
        return None

    data_folders = []
    all_bd4_files = []
    all_uvn_files = []

    # 遍历所有日期文件夹和子文件夹，收集 BD4 和 UVN 文件
    for root, _, files in os.walk(l1b_dir):
        for file in files:
            # 转换为小写后检查后缀，确保容错性
            if file.lower().endswith(".zip") or file.lower().endswith(".nc"):
                if "ra_bd4" in file.lower():
                    all_bd4_files.append(os.path.join(root, file))
                elif "ir_uvn" in file.lower():
                    all_uvn_files.append(os.path.join(root, file))

    # 日志信息
    print(f"找到 BD4 文件: {len(all_bd4_files)} 个")
    print(f"找到 UVN 文件: {len(all_uvn_files)} 个")

    if not all_bd4_files or not all_uvn_files:
        print("未找到完整的数据集")
        return None

    # 按日期分组匹配 BD4 和 UVN 文件
    for bd4_file in all_bd4_files:
        bd4_date = extract_date_from_filename(bd4_file).date()  # 只取日期部分
        for uvn_file in all_uvn_files:
            uvn_date = extract_date_from_filename(uvn_file).date()  # 只取日期部分

            if bd4_date == uvn_date:  # 同一天匹配
                subfolder_name = bd4_date.strftime('%Y%m%d')
                process_folder = os.path.join(result_dir, subfolder_name)

                if not os.path.exists(process_folder):
                    os.makedirs(process_folder)

                data_folders.append({
                    "original": os.path.dirname(bd4_file),
                    "process": process_folder,
                    "subfolder_name": subfolder_name,
                    "bd4_file": bd4_file,
                    "uvn_file": uvn_file
                })
                break  # 一旦匹配成功,跳出内层循环

    if not data_folders:
        print("未找到完整的数据集")
        return None

    print(f"找到 {len(data_folders)} 组匹配数据")
    return data_folders

def extract_date_from_filename(file_path):
    """
    从文件名中提取日期
    支持格式：YYYYMMDD 或 YYYYMMDDTHHMMSS,优先提取第一个有效日期
    返回 datetime 对象
    """
    filename = os.path.basename(file_path)

    # 匹配 YYYYMMDD 或 YYYYMMDDTHHMMSS 格式的日期
    import re
    match = re.search(r'\d{8}(?=T?\d{0,6})', filename)
    if match:
        date_str = match.group()
        return datetime.strptime(date_str, "%Y%m%d")
    else:
        raise ValueError(f"无法从文件名中提取日期: {filename}")


def extract_and_load_data(folder_info: dict, band_name: str) -> dict:
    """解压或直接加载数据（支持 zip 和 nc 格式）"""
    data = {}

    print(f"\n=== 处理{band_name}数据 ===")
    print(f"时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"用户: {os.getlogin()}")

    # 根据 band_name 选择对应的文件路径
    file_path = folder_info['bd4_file'] if band_name == 'RA_BD4' else folder_info['uvn_file']
    print(f"处理文件: {file_path}")

    # 判断文件类型，若为zip则解压，否则直接读取
    temp_nc_path = None
    if file_path.lower().endswith(".zip"):
        temp_dir = create_temp_dir(os.path.join(r"M:", "temp"), f"{band_name}")
        print(f"创建临时目录: {temp_dir}")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                nc_members = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_members:
                    print("警告：ZIP文件中未找到.nc文件")
                    return {}
                nc_file = nc_members[0]
                print(f"提取文件: {nc_file}")

                # 将解压后的文件写入临时文件
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(zip_ref.read(nc_file))
                    temp_nc_path = tmp_file.name
                print(f"解压临时文件路径: {temp_nc_path}")
            file_to_open = temp_nc_path
        except Exception as e:
            print(f"处理ZIP文件 {file_path} 时出错: {str(e)}")
            traceback.print_exc()
            return {}
    elif file_path.lower().endswith(".nc"):
        # 直接使用 nc 文件
        file_to_open = file_path
    else:
        print(f"错误: 不支持的文件格式: {file_path}")
        return {}

    try:
        with h5py.File(file_to_open, 'r') as h5:
            print("\n=== 文件结构信息 ===")
            # 可选：遍历文件结构打印数据信息
            # def print_dataset_info(name, obj):
            #     if isinstance(obj, h5py.Dataset):
            #         print(f"数据集: {name}, 形状: {obj.shape}, 类型: {obj.dtype}")
            # h5.visititems(print_dataset_info)

            # 定义变量映射
            if band_name == 'RA_BD4':
                var_mapping = {
                    'radiance': ['BAND4_RADIANCE/STANDARD_MODE/OBSERVATIONS/radiance', 'radiance'],
                    'wavelength': ['BAND4_RADIANCE/STANDARD_MODE/INSTRUMENT/nominal_wavelength', 'wavelength'],
                    'quality_level': ['BAND4_RADIANCE/STANDARD_MODE/OBSERVATIONS/quality_level', 'quality_level'],
                    'solar_zenith_angle': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/solar_zenith_angle',
                                           'solar_zenith_angle'],
                    'solar_azimuth_angle': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/solar_azimuth_angle',
                                            'solar_azimuth_angle'],
                    'viewing_zenith_angle': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/viewing_zenith_angle',
                                             'viewing_zenith_angle'],
                    'viewing_azimuth_angle': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/viewing_azimuth_angle',
                                              'viewing_azimuth_angle'],
                    'latitude': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/latitude', 'latitude'],
                    'longitude': ['BAND4_RADIANCE/STANDARD_MODE/GEODATA/longitude', 'longitude'],
                    'time': ['BAND4_RADIANCE/STANDARD_MODE/OBSERVATIONS/time', 'time'],
                    'delta_time': ['BAND4_RADIANCE/STANDARD_MODE/OBSERVATIONS/delta_time', 'delta_time']
                }
            else:  # RA_UVN
                var_mapping = {
                    'irradiance': ['BAND4_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/irradiance', 'irradiance'],
                    'irradiance_error': ['BAND4_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/irradiance_error',
                                         'irradiance_error'],
                    'irradiance_noise': ['BAND4_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/irradiance_noise',
                                         'irradiance_noise'],
                    'wavelength': ['BAND4_IRRADIANCE/STANDARD_MODE/INSTRUMENT/nominal_wavelength', 'wavelength'],
                    'earth_sun_distance': ['BAND4_IRRADIANCE/STANDARD_MODE/GEODATA/earth_sun_distance',
                                           'earth_sun_distance']
                }

            # 读取数据
            result = {}
            for var_name, (path, target_name) in var_mapping.items():
                try:
                    if path in h5:
                        data_array = h5[path][:]
                        result[target_name] = data_array
                    else:
                        print(f"警告: 找不到路径 {path}")
                except Exception as e:
                    print(f"读取变量 {var_name} 时出错: {str(e)}")
            data[os.path.basename(file_path)] = result

    except Exception as e:
        print(f"处理文件 {file_to_open} 时出错: {str(e)}")
        traceback.print_exc()
        return {}
    finally:
        # 如果使用了临时文件，则清理该临时文件
        if temp_nc_path and os.path.exists(temp_nc_path):
            os.remove(temp_nc_path)
        # 如果是zip分支创建的临时目录，也清理之
        if file_path.lower().endswith(".zip"):
            temp_dir = os.path.dirname(temp_nc_path) if temp_nc_path else None
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return data

def save_as_netcdf(radiance, wavelength_standard, latitude, longitude,
                   solar_zenith_angle, solar_azimuth_angle,
                   viewing_zenith_angle, viewing_azimuth_angle,
                   delta_time, time,
                   irradiance, irradiance_error, irradiance_noise,
                   folder_path, subfolder_name):
    """
    保存数据为 NetCDF 文件。

    Args:
        radiance: 辐射率数据，NumPy 数组，形状 (time, scanline, ground_pixel)。
        wavelength_standard: 波长标准，NumPy 数组，形状 (scanline, ground_pixel)。
        latitude: 纬度数据，NumPy 数组，形状 (time, scanline)。
        longitude: 经度数据，NumPy 数组，形状 (time, scanline)。
        solar_zenith_angle: 太阳天顶角，NumPy 数组，形状 (time, scanline)。
        solar_azimuth_angle: 太阳方位角，NumPy 数组，形状 (time, scanline)。
        viewing_zenith_angle: 观测天顶角，NumPy 数组，形状 (time, scanline)。
        viewing_azimuth_angle: 观测方位角，NumPy 数组，形状 (time, scanline)。
        delta_time: 时间增量，NumPy 数组，形状 (time,) 或 (1, time)。
        time: 基准时间，数值或 NumPy 标量。
        irradiance: 辐照度数据，NumPy 数组，形状与 radiance 相同。
        irradiance_error: 辐照度误差数据，NumPy 数组，形状与 radiance 相同。
        irradiance_noise: 辐照度噪声数据，NumPy 数组，形状与 radiance 相同。
        folder_path: 输出文件夹路径。
        subfolder_name: 子文件夹名称。

    Returns:
        True: 保存成功。
        False: 保存失败。
    """
    try:
        # 构建输出文件名
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(folder_path, f"{subfolder_name}_{timestamp}_result.nc")
        print(f"\n准备保存数据到: {output_file}")

        # 数据类型和形状检查
        print("\n=== 数据维度检查 ===")
        print(f"radiance: shape={radiance.shape}, dtype={radiance.dtype}")
        print(f"latitude: shape={latitude.shape}, dtype={latitude.dtype}")
        print(f"longitude: shape={longitude.shape}, dtype={longitude.dtype}")
        print(f"solar_zenith_angle: shape={solar_zenith_angle.shape}, dtype={solar_zenith_angle.dtype}")
        print(f"solar_azimuth_angle: shape={solar_azimuth_angle.shape}, dtype={solar_azimuth_angle.dtype}")
        print(f"viewing_zenith_angle: shape={viewing_zenith_angle.shape}, dtype={viewing_zenith_angle.dtype}")
        print(f"viewing_azimuth_angle: shape={viewing_azimuth_angle.shape}, dtype={viewing_azimuth_angle.dtype}")
        print(f"wavelength_standard: shape={wavelength_standard.shape}, dtype={wavelength_standard.dtype}")
        print(f"delta_time: shape={delta_time.shape}, dtype={delta_time.dtype}")
        print(f"irradiance: shape={irradiance.shape}, dtype={irradiance.dtype}")
        print(f"irradiance_error: shape={irradiance_error.shape}, dtype={irradiance_error.dtype}")
        print(f"irradiance_noise: shape={irradiance_noise.shape}, dtype={irradiance_noise.dtype}")

        # 处理单时间步长的情况，并统一 delta_time 的形状
        if radiance.shape[0] == 1:
            radiance = radiance[0]
            latitude = latitude[0]
            longitude = longitude[0]
            solar_zenith_angle = solar_zenith_angle[0]
            solar_azimuth_angle = solar_azimuth_angle[0]
            viewing_zenith_angle = viewing_zenith_angle[0]
            viewing_azimuth_angle = viewing_azimuth_angle[0]
            wavelength_standard = wavelength_standard[0]
            delta_time = delta_time.flatten()  # 统一 delta_time 为一维数组
            irradiance = irradiance[0]
            irradiance_error = irradiance_error[0]
            irradiance_noise = irradiance_noise[0]
        elif delta_time.ndim == 2 and delta_time.shape[0] == 1:
            delta_time = delta_time[0]

        # 获取维度信息
        num_tracks, num_pixels, num_wavelengths = radiance.shape
        print(f"数据维度: tracks={num_tracks}, pixels={num_pixels}, wavelengths={num_wavelengths}")

        # 创建 NetCDF 文件
        with Dataset(output_file, "w", format="NETCDF4") as ncfile:
            # 定义维度
            ncfile.createDimension("time", num_tracks)
            ncfile.createDimension("scanline", num_pixels)
            ncfile.createDimension("ground_pixel", num_wavelengths)

            # 定义变量（原有变量）
            radiance_var = ncfile.createVariable("radiance", np.float32, ("time", "scanline", "ground_pixel"), zlib=True, complevel=5)
            wavelength_var = ncfile.createVariable("wavelength", np.float32, ("scanline", "ground_pixel"))
            latitude_var = ncfile.createVariable("latitude", np.float32, ("time", "scanline"))
            longitude_var = ncfile.createVariable("longitude", np.float32, ("time", "scanline"))
            solar_zenith_var = ncfile.createVariable("solar_zenith_angle", np.float32, ("time", "scanline"))
            solar_azimuth_var = ncfile.createVariable("solar_azimuth_angle", np.float32, ("time", "scanline"))
            viewing_zenith_var = ncfile.createVariable("viewing_zenith_angle", np.float32, ("time", "scanline"))
            viewing_azimuth_var = ncfile.createVariable("viewing_azimuth_angle", np.float32, ("time", "scanline"))
            delta_time_var = ncfile.createVariable("delta_time", np.float64, ("time",))
            time_utc_var = ncfile.createVariable("time_utc", "S26", ("time",))

            # 新增 UVN 变量：irradiance, irradiance_error, irradiance_noise
            irradiance_var = ncfile.createVariable("irradiance", np.float32, ("time", "scanline", "ground_pixel"), zlib=True, complevel=5)
            irradiance_error_var = ncfile.createVariable("irradiance_error", np.float32, ("time", "scanline", "ground_pixel"), zlib=True, complevel=5)
            irradiance_noise_var = ncfile.createVariable("irradiance_noise", np.float32, ("time", "scanline", "ground_pixel"), zlib=True, complevel=5)

            # 赋值变量
            radiance_var[:] = radiance
            wavelength_var[:] = wavelength_standard
            latitude_var[:] = latitude
            longitude_var[:] = longitude
            solar_zenith_var[:] = solar_zenith_angle
            solar_azimuth_var[:] = solar_azimuth_angle
            viewing_zenith_var[:] = viewing_zenith_angle
            viewing_azimuth_var[:] = viewing_azimuth_angle
            delta_time_var[:] = delta_time
            irradiance_var[:] = irradiance
            irradiance_error_var[:] = irradiance_error
            irradiance_noise_var[:] = irradiance_noise

            # 处理时间变量
            base_time = datetime(2010, 1, 1) + timedelta(seconds=float(time.item()))
            print(f"基准时间: {base_time}")

            time_values = np.empty(num_tracks, dtype="S26")
            print("\n=== Delta Time 调试信息 ===")
            print(f"Delta time 前三个值: {delta_time[:3]}")
            for i in range(num_tracks):
                delta_seconds = float(delta_time[i]) / 1000.0
                current_time = base_time + timedelta(seconds=delta_seconds)
                time_values[i] = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3].encode('ascii')
            print("\n=== 处理后的 UTC 时间验证 ===")
            for i, time_val in enumerate(time_values):
                print(f"[{i}] {time_val.decode('ascii')}")
            time_utc_var[:] = time_values

            # 添加变量属性
            radiance_var.long_name = "Radiance"
            radiance_var.units = "W/m^2/sr/nm"
            wavelength_var.long_name = "Wavelength"
            wavelength_var.units = "nm"
            latitude_var.long_name = "Latitude"
            latitude_var.units = "degrees_north"
            longitude_var.long_name = "Longitude"
            longitude_var.units = "degrees_east"
            solar_zenith_var.long_name = "Solar Zenith Angle"
            solar_zenith_var.units = "degrees"
            solar_azimuth_var.long_name = "Solar Azimuth Angle"
            solar_azimuth_var.units = "degrees"
            viewing_zenith_var.long_name = "Viewing Zenith Angle"
            viewing_zenith_var.units = "degrees"
            viewing_azimuth_var.long_name = "Viewing Azimuth Angle"
            viewing_azimuth_var.units = "degrees"
            time_utc_var.long_name = "Time UTC"
            time_utc_var.units = "ISO 8601 string"

            irradiance_var.long_name = "Irradiance"
            irradiance_var.units = "W/m^2/sr/nm"  # 请根据实际单位进行修改
            irradiance_error_var.long_name = "Irradiance Error"
            irradiance_error_var.units = "W/m^2/sr/nm"  # 请根据实际单位进行修改
            irradiance_noise_var.long_name = "Irradiance Noise"
            irradiance_noise_var.units = "W/m^2/sr/nm"  # 请根据实际单位进行修改

            # 添加全局属性
            ncfile.description = "Processed TROPOMI L1B Data"
            ncfile.source = "TROPOMI L1B Processing Pipeline"
            ncfile.creation_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            ncfile.created_by = os.getlogin()
            ncfile.processing_software_version = "1.0"

        print(f"数据成功保存到: {output_file}")
        return True

    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """
    主程序：处理 TROPOMI L1B 数据
    每组数据单独处理并保存为单独的 NetCDF 文件
    """
    print("=== 开始处理 TROPOMI L1B 数据 ===")
    print(f"开始时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    try:
        # 1. 查找所有数据文件夹
        all_folders = find_data_folder()
        if not all_folders:
            print("未找到任何数据文件夹")
            return

        print(f"找到 {len(all_folders)} 组数据待处理")

        # 按日期对数据进行分组
        date_groups = {}
        for folder_info in all_folders:
            date = extract_date_from_filename(folder_info['bd4_file']).date()
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(folder_info)

        # 2. 遍历处理每天的数据
        for date, folders in date_groups.items():
            print(f"\n开始处理 {date} 的数据，共 {len(folders)} 组")
            uvn_file = folders[0]['uvn_file']  # 该天的UVN文件（所有BD4共用）
            process_success = True  # 标记该天的所有数据是否都处理成功

            # 处理该天的每组数据
            for folder_idx, folder_info in enumerate(folders, 1):
                print(f"\n处理第 {folder_idx}/{len(folders)} 组数据")

                try:
                    # 2.1 加载 Band 4 数据
                    band_data = extract_and_load_data(folder_info, "RA_BD4")
                    if not band_data:
                        print("跳过当前组：Band 4 数据加载失败")
                        process_success = False
                        continue

                    # 2.2 加载辐照度数据
                    irradiance_data = extract_and_load_data(folder_info, "RA_UVN")
                    if not irradiance_data:
                        print("跳过当前组：辐照度数据加载失败")
                        process_success = False
                        continue

                    # 2.3 提取所需变量
                    first_band_file = list(band_data.values())[0]
                    radiance = first_band_file.get("radiance", None)
                    latitude = first_band_file.get("latitude", None)
                    longitude = first_band_file.get("longitude", None)
                    solar_zenith_angle = first_band_file.get("solar_zenith_angle", None)
                    solar_azimuth_angle = first_band_file.get("solar_azimuth_angle", None)
                    viewing_zenith_angle = first_band_file.get("viewing_zenith_angle", None)
                    viewing_azimuth_angle = first_band_file.get("viewing_azimuth_angle", None)
                    delta_time = first_band_file.get("delta_time", None)
                    time = first_band_file.get("time", None)

                    # 同时提取 UVN 数据中的 irradiance 变量及其误差和噪声
                    uvn_first_file = list(irradiance_data.values())[0]
                    irradiance = uvn_first_file.get("irradiance", None)
                    irradiance_error = uvn_first_file.get("irradiance_error", None)
                    irradiance_noise = uvn_first_file.get("irradiance_noise", None)

                    # 数据完整性检查
                    missing_vars = [name for name, var in zip(
                        ["radiance", "latitude", "longitude", "solar_zenith_angle", "solar_azimuth_angle",
                         "viewing_zenith_angle", "viewing_azimuth_angle", "delta_time", "time",
                         "irradiance", "irradiance_error", "irradiance_noise"],
                        [radiance, latitude, longitude, solar_zenith_angle, solar_azimuth_angle,
                         viewing_zenith_angle, viewing_azimuth_angle, delta_time, time,
                         irradiance, irradiance_error, irradiance_noise]) if var is None]
                    if missing_vars:
                        print(f"跳过当前组：缺少必要变量 {missing_vars}")
                        process_success = False
                        continue

                    # 2.4 保存处理结果
                    wavelength_standard = first_band_file.get("wavelength", None)
                    if wavelength_standard is None:
                        print("跳过当前组：缺少波长数据")
                        process_success = False
                        continue

                    success = save_as_netcdf(
                        radiance,
                        wavelength_standard,
                        latitude,
                        longitude,
                        solar_zenith_angle,
                        solar_azimuth_angle,
                        viewing_zenith_angle,
                        viewing_azimuth_angle,
                        delta_time,
                        time,
                        irradiance,
                        irradiance_error,
                        irradiance_noise,
                        folder_info['process'],
                        folder_info['subfolder_name']
                    )

                    if success:
                        print(f"第 {folder_idx}/{len(folders)} 组数据处理完成")
                        # 删除处理完成的BD4文件
                        if os.path.exists(folder_info['bd4_file']):
                            os.remove(folder_info['bd4_file'])
                            print(f"已删除BD4文件: {folder_info['bd4_file']}")
                    else:
                        print(f"第 {folder_idx}/{len(folders)} 组数据处理失败")
                        process_success = False

                except Exception as e:
                    print(f"处理组 {folder_info['original']} 时出错: {str(e)}")
                    process_success = False
                    continue

            # 该天所有数据处理完成后，如果全部成功则删除UVN文件
            if process_success and os.path.exists(uvn_file):
                os.remove(uvn_file)
                print(f"已删除UVN文件: {uvn_file}")

        print("=== 数据处理完成 ===")

    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")

    finally:
        print(f"结束时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行失败: {str(e)}")
        traceback.print_exc()