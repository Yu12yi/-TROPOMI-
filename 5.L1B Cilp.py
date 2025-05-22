import xarray as xr
import numpy as np
import os
import shutil
from datetime import datetime

def validate_dimensions(ds):
    """验证数据集维度"""
    required_dims = {
        'time', 'scanline', 'ground_pixel'
    }

    if not all(dim in ds.dims for dim in required_dims):
        return False
    return True

def clip_us_region_and_replace(input_file, output_file):
    """
    裁剪NetCDF文件到美国区域，保存裁剪结果并删除原始文件。

    参数:
        input_file (str): 输入 NetCDF 文件的完整路径。
        output_file (str): 裁剪后 NetCDF 文件的完整路径。

    返回值:
        bool: 裁剪并替换成功返回 True，失败返回 False。
    """
    # 美国区域范围（包括本土和阿拉斯加，可根据实际情况调整）
    US_LON_MIN, US_LON_MAX = -172.4543, -66.9548  # 扩大经度范围以包含阿拉斯加和更广区域
    US_LAT_MIN, US_LAT_MAX = 18.9100, 71.3866   # 扩大纬度范围以包含更广区域
    print(f"处理文件: {input_file}")

    try:
        ds = xr.open_dataset(input_file)
        if not validate_dimensions(ds):
            print(f"文件 {input_file} 缺少必要维度，跳过裁剪。")
            return False

        # 创建区域掩码
        lat_mask = (ds.latitude >= US_LAT_MIN) & (ds.latitude <= US_LAT_MAX)
        lon_mask = (ds.longitude >= US_LON_MIN) & (ds.longitude <= US_LON_MAX)
        region_mask = lat_mask & lon_mask

        valid_times, valid_scanlines = np.where(region_mask)
        if len(valid_times) == 0:
            print(f"{input_file} 无有效区域数据，跳过保存。")
            return False

        # 获取唯一的time和scanline索引
        unique_times = np.unique(valid_times)
        unique_scanlines = np.unique(valid_scanlines)

        clipped_ds = xr.Dataset()

        # 处理不同维度的变量
        for var_name, var in ds.variables.items():
            if var_name in ds.coords:
                continue

            dims = var.dims
            if 'time' in dims and 'scanline' in dims and 'ground_pixel' in dims:
                clipped = var.isel(time=unique_times, scanline=unique_scanlines)
            elif 'time' in dims and 'scanline' in dims:
                clipped = var.isel(time=unique_times, scanline=unique_scanlines)
            elif 'time' in dims:
                clipped = var.isel(time=unique_times)
            elif 'scanline' in dims and 'ground_pixel' in dims:
                clipped = var.isel(scanline=unique_scanlines)
            else:
                clipped = var
            clipped_ds[var_name] = clipped

        # 确保维度一致性
        clipped_ds['time'] = ds['time'].isel(time=unique_times)
        clipped_ds['scanline'] = np.arange(len(unique_scanlines))
        if 'ground_pixel' in ds.dims:
            clipped_ds['ground_pixel'] = ds['ground_pixel']

        # 更新元数据
        clipped_ds.attrs = {
            **ds.attrs,
            'preprocessing': '美国区域裁剪',
            'lon_range': f"{US_LON_MIN}~{US_LON_MAX}",
            'lat_range': f"{US_LAT_MIN}~{US_LAT_MAX}",
            'clip_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 配置压缩编码
        encoding = {}
        for var in clipped_ds.variables:
            if var == 'time_utc': # 针对 time_utc 变量，移除压缩设置
                encoding[var] = {
                    '_FillValue': None # 字符串类型通常不需要 _FillValue
                }
            else: # 其他变量保持原有的压缩设置
                encoding[var] = {
                    'zlib': True,
                    'complevel': 5,
                    '_FillValue': None if var in ['time', 'scanline', 'ground_pixel'] else np.nan
                }

        clipped_ds.to_netcdf(output_file, encoding=encoding)
        ds.close()
        clipped_ds.close()

        print(f"已保存裁剪文件: {output_file}")

        # 删除原始文件
        try:
            os.remove(input_file)
            print(f"已删除原始文件: {input_file}")
        except Exception as delete_err:
            print(f"删除原始文件 {input_file} 失败: {delete_err}")
            return False # 删除失败也返回 False，表示整体操作不完全成功

        return True

    except Exception as e:
        print(f"处理文件 {input_file} 时发生错误: {e}")
        return False

if __name__ == "__main__":
    root_dir = r"M:\TROPOMI_S5P\temp\result"  # 根目录
    date_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.isdigit() and len(f) == 8] # 筛选日期文件夹

    if not date_folders:
        print(f"在 {root_dir} 目录下未找到日期命名的文件夹。")
    else:
        print(f"找到以下日期文件夹: {date_folders}")

    for date_folder in date_folders:
        input_dir = os.path.join(root_dir, date_folder)
        output_dir = input_dir # 裁剪后的文件保存在同一日期文件夹下

        nc_files = [f for f in os.listdir(input_dir) if f.endswith('.nc')]
        if not nc_files:
            print(f"在 {input_dir} 目录下未找到NC文件，跳过。")
            continue

        print(f"\n开始处理日期文件夹: {date_folder}")
        for nc_file in nc_files:
            input_path = os.path.join(input_dir, nc_file)
            output_path = os.path.join(output_dir, nc_file.replace('.nc', '_clipped.nc')) # 输出文件名添加 "_clipped" 后缀

            success = clip_us_region_and_replace(input_path, output_path)
            if success:
                print(f"  {nc_file} 裁剪并替换完成。")
            else:
                print(f"  {nc_file} 裁剪或替换过程中出现错误。")

    print("\n所有日期文件夹处理完成。")