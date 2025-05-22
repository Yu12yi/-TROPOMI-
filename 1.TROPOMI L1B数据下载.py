import os
import requests
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta
import json


class SentinelDownload:
    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password
        self.token = self.get_access_token()

        # 优化会话设置
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,  # 单线程模式下保持1个连接池
            pool_maxsize=5,  # 但允许最多5个连接以处理重定向
            max_retries=3,  # 内置重试机制
            pool_block=False
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def get_access_token(self) -> str:
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.user_name,
            "password": self.password,
            "grant_type": "password",
        }
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(token_url, data=data)
                response.raise_for_status()
                print("Token 获取成功！")
                return response.json()["access_token"]
            except Exception as e:
                print(f"获取 Token 时发生错误: {e}")
                if attempt == retries - 1:
                    raise Exception("获取 Token 失败，已达到最大重试次数")
                print(f"等待5秒后进行第 {attempt + 2} 次重试...")
                time.sleep(5)

    def search(self, search_url: str) -> list:
        """
        检索 Sentinel 数据，处理分页，支持token过期后继续检索
        """
        search_results = []
        page_size = 50  # 每页返回记录数
        skip = 0  # 起始偏移量
        retries = 3  # 最大重试次数

        while True:
            paginated_url = f"{search_url}&$top={page_size}&$skip={skip}"

            for attempt in range(retries):
                try:
                    response = self.session.get(
                        paginated_url,
                        headers={
                            "Authorization": f"Bearer {self.token}",
                            "Connection": "keep-alive",
                            "Accept-Encoding": "gzip, deflate"
                        },
                        timeout=30
                    )

                    # 处理token过期情况
                    if response.status_code == 401 or response.status_code == 403:
                        print(f"Token已过期或访问被拒绝，正在重新获取（第 {attempt + 1}/{retries} 次）...")
                        self.token = self.get_access_token()
                        continue

                    # 处理429错误(太多请求)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 30))
                        print(f"服务器请求限流(429)，等待{retry_after}秒后重试...")
                        time.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    json_info = response.json()

                    current_page_results = json_info.get("value", [])
                    if not current_page_results:  # 如果当前页面没有数据，说明已经到达末尾
                        print("已加载所有数据")
                        print(f"共检索到 {len(search_results)} 条数据")
                        return search_results

                    search_results.extend(current_page_results)
                    skip += page_size  # 更新偏移量，获取下一页
                    print(f"已加载 {len(search_results)} 条记录...")

                    # 若已经加载的结果达到上限，直接返回
                    if len(search_results) >= 1000:  # 可根据实际需求设置上限
                        print(f"检索到足够数据 ({len(search_results)} 条)，停止继续加载。")
                        return search_results

                    # 添加短暂延时，避免请求过于频繁
                    time.sleep(1)

                    break  # 成功获取数据，跳出重试循环

                except Exception as e:
                    print(f"检索数据时发生错误: {e}")
                    if attempt == retries - 1:  # 最后一次重试失败
                        print("达到最大重试次数，返回已获取的数据")
                        return search_results
                    print(f"2秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(2)

    def download_file(self, product_id: str, save_path: str):
        """优化的单线程下载方法，提高下载速度"""
        # 检查原始 .zip 文件是否存在
        if os.path.exists(f"{save_path}.zip"):
            print(f"{save_path}.zip 已存在，跳过下载...")
            return True

        # 检查用户可能重命名后的 .nc 文件是否存在
        base_name = os.path.basename(save_path)
        if "_20" in base_name:  # 检查是否有日期后缀
            # 提取可能的原始文件名，去除日期后缀
            parts = base_name.split("_")
            date_suffix = parts[-1]  # 假设最后一部分是日期
            original_parts = parts[:-1]  # 除了日期外的所有部分

            if ".nc" not in "_".join(original_parts):
                original_parts[-1] += ".nc"

            original_filename = "_".join(original_parts)
            original_path = os.path.join(os.path.dirname(save_path), original_filename)

            if os.path.exists(original_path):
                print(f"已重命名的文件 {original_filename} 已存在，跳过下载...")
                return True

        # 检查是否有部分下载的文件
        if os.path.exists(f"{save_path}.tmp"):
            # 获取已下载大小
            downloaded_size = os.path.getsize(f"{save_path}.tmp")
            resume_header = {'Range': f'bytes={downloaded_size}-'}
            print(f"发现已下载的部分文件({downloaded_size}字节)，将尝试继续下载...")
            mode = "ab"  # 追加模式
        else:
            downloaded_size = 0
            resume_header = {}
            mode = "wb"  # 新建模式

        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        retries = 5

        # 优化的请求头
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "SentinelDownloader/1.0"
        }
        headers.update(resume_header)

        # 增大chunk_size以减少请求次数，提高吞吐量
        chunk_size = 1024 * 1024 * 64  # 64MB

        # 优化进度条更新频率
        progress_update_interval = 1024 * 1024 * 20  # 每4MB更新一次进度条

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # 使用临时变量存储当前已下载大小，减少进度条更新频率
        current_progress = downloaded_size
        last_update = downloaded_size

        for attempt in range(retries):
            try:
                # 优化的请求参数
                response = self.session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(10, 300),  # 连接超时10秒，读取超时300秒
                    allow_redirects=True
                )

                # 处理token过期情况
                if response.status_code == 401:
                    print(f"401 错误：尝试重新获取 Token（第 {attempt + 1}/{retries} 次）...")
                    self.token = self.get_access_token()
                    headers["Authorization"] = f"Bearer {self.token}"
                    continue

                # 处理429错误(太多请求)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    print(f"服务器请求限流(429)，等待{retry_after}秒后重试...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                # 处理Range请求的特殊情况
                if downloaded_size > 0 and response.status_code == 206:  # 206是部分内容的状态码
                    content_range = response.headers.get('Content-Range', '')
                    total_size = int(content_range.split('/')[-1]) if '/' in content_range else 0
                else:
                    total_size = int(response.headers.get("Content-Length", 0)) + downloaded_size

                # 创建进度条
                progress_bar = tqdm(
                    total=total_size,
                    initial=downloaded_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f"下载 {os.path.basename(save_path)}"
                )

                # 优化IO操作 - 使用带缓冲的写入
                with open(f"{save_path}.tmp", mode, buffering=8 * 1024 * 1024) as f:  # 8MB缓冲区
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            current_progress += len(chunk)

                            # 减少进度条更新频率，提高性能
                            if current_progress - last_update >= progress_update_interval:
                                progress_bar.update(current_progress - last_update)
                                last_update = current_progress

                # 确保最后更新一次进度条
                if current_progress > last_update:
                    progress_bar.update(current_progress - last_update)

                progress_bar.close()

                # 检查文件完整性
                if current_progress >= total_size:
                    # 直接使用os.rename而不是重新读写文件
                    os.rename(f"{save_path}.tmp", f"{save_path}.zip")
                    print(f"\n{save_path}.zip 下载完成！")
                    return True
                else:
                    print(f"\n文件下载不完整({current_progress}/{total_size}字节)，将重试")
                    # 不删除临时文件，下次可以继续下载

            except requests.exceptions.ReadTimeout:
                # 特别处理读取超时，可能只是服务器响应慢
                print(f"\n下载 {save_path} 时读取超时，但可能部分数据已下载。将在5秒后重试...")
                time.sleep(5)
            except requests.exceptions.ChunkedEncodingError:
                # 处理分块编码错误，这通常是因为连接中断
                print(f"\n下载 {save_path} 时连接中断，部分数据可能已下载。将在5秒后重试...")
                time.sleep(5)
            except Exception as e:
                print(f"\n下载 {save_path} 时发生错误: {e}")
                if attempt < retries - 1:
                    wait_time = min(30, 10 * (attempt + 1))  # 使用指数退避策略
                    print(f"等待{wait_time}秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(wait_time)
                else:
                    print(f"下载 {save_path} 失败，已达到最大重试次数。")
                    return False

        return False

    def download_all(self, search_url: str, save_folder: str, data_type: str, max_per_day: int, max_workers=2):
        """
        下载所有检索结果，使用适度的并行下载
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        search_results = self.search(search_url)

        # 创建日志文件
        log_file = os.path.join(save_folder, 'download_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"下载开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最大每日下载限制: {max_per_day}\n\n")
            f.write("数据统计信息:\n")
            f.write("-" * 50 + "\n")

        # 按日期分组
        daily_results = {}
        for item in search_results:
            date_str = item["ContentDate"]["Start"].split("T")[0]
            if date_str not in daily_results:
                daily_results[date_str] = []
            daily_results[date_str].append(item)

        # 记录所有日期的数据情况
        start_date = min(daily_results.keys()) if daily_results else None
        end_date = max(daily_results.keys()) if daily_results else None

        if start_date and end_date:
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                files_count = len(daily_results.get(date_str, []))

                with open(log_file, 'a', encoding='utf-8') as f:
                    if files_count == 0:
                        f.write(f"{date_str}: 未检索到数据\n")
                        print(f"{date_str}: 未检索到数据")
                    else:
                        f.write(f"{date_str}: 检索到 {files_count} 个文件")
                        if files_count > max_per_day:
                            f.write(" (超过每日限制)\n")
                            print(f"{date_str} 的文件数量（{files_count}）超过最大限制（{max_per_day}），跳过")
                        else:
                            f.write("\n")
                            print(f"{date_str} 检索到 {files_count} 个文件，开始下载...")

                current_date += timedelta(days=1)

        # 收集所有需要下载的文件信息
        files_to_download = []
        for date, items in sorted(daily_results.items()):
            if len(items) > max_per_day:
                continue

            for item in items:
                files_to_download.append((item["Id"], item["Name"], date))

        # 按照需求使用单线程模式下载
        if files_to_download:
            # 使用max_workers=1确保单线程模式或根据用户设置
            with ThreadPoolExecutor(max_workers=1) as executor:  # 强制设为1确保单线程
                futures = []
                for product_id, file_name, date in files_to_download:
                    save_path = os.path.join(save_folder, f"{file_name}_{date}")
                    future = executor.submit(self.download_file, product_id, save_path)
                    futures.append((future, file_name, date))
                    # 添加延时避免同时启动太多下载任务
                    time.sleep(1)

                # 处理下载结果
                for future, file_name, date in futures:
                    try:
                        result = future.result()
                        if result:
                            print(f"文件 {file_name}_{date} 下载成功")
                        else:
                            print(f"文件 {file_name}_{date} 下载失败")
                    except Exception as e:
                        print(f"文件 {file_name}_{date} 下载异常: {e}")
        else:
            print("没有找到符合条件的文件需要下载")

        # 记录下载完成信息
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"下载结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def start_search():
    # 获取用户输入的参数
    start_date_str = start_date_var.get()
    end_date_str = end_date_var.get()
    data_names_input = data_name_var.get()  # 支持输入单个或多个数据名称，多个名称以英文逗号分隔
    data_type = data_type_var.get()
    max_per_day = int(max_per_day_var.get())
    max_workers = int(max_workers_var.get())  # 获取用户设置的最大线程数
    save_folder = filedialog.askdirectory(title="选择保存目录")
    if not save_folder:
        messagebox.showerror("错误", "请选择保存目录！")
        return

    # 将输入的日期转换为 datetime 对象
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("错误", "日期格式不正确，请使用YYYY-MM-DD格式！")
        return

    # 将用户输入的名称拆分，过滤空白项
    data_names = [dn.strip() for dn in data_names_input.split(",") if dn.strip()]
    if not data_names:
        messagebox.showerror("错误", "请输入至少一个数据名称！")
        return

    # 根据每个数据名称构造条件，针对特定数据类型添加地理位置过滤条件
    name_conditions = []
    for dn in data_names:
        # 对BD4数据添加北美地区的地理位置限制
        if (dn == "L1B_RA_BD4"):
            condition = (
                f"(contains(Name,'{dn}') and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON (("
                f"-130.13 26.53, -66.12 26.53, -66.12 48.71, -130.13 48.71, -130.13 26.53))'))"
            )
        # 保持原有的NO2数据的地理位置限制
        elif (dn == "L2__NO2___"):
            condition = (
                f"(contains(Name,'{dn}') and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON ((-128.99 26.05, -63.54 26.05, "
                f"-63.54 52.81, -128.99 52.81, -128.99 26.05))'))"
            )
        # 其他数据类型不添加地理位置限制
        else:
            condition = f"contains(Name,'{dn}')"
        name_conditions.append(condition)
    name_filter = "(" + " or ".join(name_conditions) + ")"

    # 显示下载进度窗口
    progress_window = tk.Toplevel(root)
    progress_window.title("下载进度")
    progress_window.geometry("400x200")
    progress_label = tk.Label(progress_window, text="正在准备下载...")
    progress_label.pack(pady=20)
    progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
    progress_bar.pack(fill=tk.X, padx=20, pady=10)
    progress_bar.start()

    def download_thread():
        try:
            # 初始化下载器
            downloader = SentinelDownload(USER_NAME, PASSWORD)

            current_start_date = start_date
            while current_start_date < end_date:
                # 获取当前月的起始和结束日期
                current_end_date = (current_start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                if current_end_date > end_date:
                    current_end_date = end_date

                # 根据数据类型构造附加过滤条件
                type_filter = ""
                if data_type != "全部":
                    type_filter = f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingMode' and att/Value eq '{data_type}') "

                # 使用新的 URL 模板，包含动态日期及附加参数
                search_url = (
                    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
                    f"&$filter=((Collection/Name eq 'SENTINEL-5P' and "
                    f"(Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'instrumentShortName' and "
                    f"att/OData.CSC.StringAttribute/Value eq 'TROPOMI') and "
                    f"{name_filter}) and Online eq true) and "
                    f"ContentDate/Start ge {current_start_date.strftime('%Y-%m-%dT00:00:00.000Z')} and "
                    f"ContentDate/Start lt {current_end_date.strftime('%Y-%m-%dT23:59:59.999Z')}"
                    f"{type_filter}"
                    f")&$orderby=ContentDate/Start desc&$expand=Attributes&$count=True&$top=50&$expand=Assets&$skip=0"
                )

                # 下载当前月的数据
                downloader.download_all(search_url, save_folder, data_type, max_per_day, 1)  # 强制使用单线程

                # 更新当前月的起始日期到下一个月
                current_start_date = current_end_date + timedelta(days=1)

            # 下载完成后的提示
            progress_window.after(0, lambda: progress_label.config(text="下载完成！"))
            progress_window.after(0, progress_bar.stop)
            messagebox.showinfo("完成", "所有文件下载完成！")
            progress_window.after(1000, progress_window.destroy)
        except Exception as e:
            progress_window.after(0, lambda: progress_label.config(text=f"下载出错: {str(e)}"))
            progress_window.after(0, progress_bar.stop)
            messagebox.showerror("错误", f"下载过程中发生错误: {str(e)}")
            progress_window.after(1000, progress_window.destroy)

    # 启动下载线程
    download_thread = threading.Thread(target=download_thread)
    download_thread.daemon = True
    download_thread.start()

# 主程序
if __name__ == "__main__":
    USER_NAME = "1254047197@qq.com"  # 替换为你的用户名
    PASSWORD = "Why23056510."  # 替换为你的密码

    root = tk.Tk()
    root.title("Sentinel 数据下载器 (单线程优化版)")

    tk.Label(root, text="开始日期 (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    start_date_var = tk.StringVar(value="2023-10-21")
    tk.Entry(root, textvariable=start_date_var).grid(row=0, column=1, padx=10, pady=5, sticky="we")

    tk.Label(root, text="结束日期 (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    end_date_var = tk.StringVar(value="2023-10-31")
    tk.Entry(root, textvariable=end_date_var).grid(row=1, column=1, padx=10, pady=5, sticky="we")

    tk.Label(root, text="数据名称:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    data_name_var = tk.StringVar(value="L1B_RA_BD4,L1B_IR_UVN")
    data_name_options = ["L1B_IR_SIR", "L1B_IR_UVN", "L1B_RA_BD1", "L1B_RA_BD2", "L1B_RA_BD3", "L1B_RA_BD4",
                         "L1B_RA_BD5",
                         "L1B_RA_BD6", "L1B_RA_BD7", "L1B_RA_BD8", "L2__NO2___", "L2_HCHO___", "L2_CLOUD___",
                         "L2_AER_AI", "L2_AER_LH"]
    ttk.Combobox(root, textvariable=data_name_var, values=data_name_options).grid(row=2, column=1, padx=10, pady=5,
                                                                                  sticky="we")

    tk.Label(root, text="数据类型:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    data_type_var = tk.StringVar(value="全部")
    ttk.Combobox(root, textvariable=data_type_var, values=["NRTI", "RPRO", "OFFL", "全部"]).grid(row=3, column=1,
                                                                                                 padx=10, pady=5,
                                                                                                 sticky="we")

    tk.Label(root, text="每天最多下载数据数量:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
    max_per_day_var = tk.StringVar(value="10")
    tk.Entry(root, textvariable=max_per_day_var).grid(row=4, column=1, padx=10, pady=5, sticky="we")

    tk.Label(root, text="最大并行下载线程数:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
    max_workers_var = tk.StringVar(value="1")  # 默认设为1以符合单线程要求
    tk.Entry(root, textvariable=max_workers_var).grid(row=5, column=1, padx=10, pady=5, sticky="we")

    tk.Button(root, text="开始下载", command=start_search, width=20).grid(row=6, column=0, columnspan=2, pady=20)

    # 优化窗口显示
    root.columnconfigure(1, weight=1)
    root.geometry("450x350")

    root.mainloop()