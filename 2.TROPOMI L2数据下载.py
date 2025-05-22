# # # _*_coding:utf-8_*_
# # # created by cy on 2023/11/12
# # # 公众号:小y只会写bug
# # # CSDN主页:https://blog.csdn.net/weixin_64989228?spm=1000.2115.3001.5343
# #
# # # changed by cy on 2024/11/26
# # # 改进：haoyiwang156@gmail.com

import os
import requests
import time
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta

class SentinelDownload:
    def __init__(self, user_name, password):
        self.user_name = user_name  # 用户名
        self.password = password  # 密码
        self.token = self.get_access_token()  # 获取Access Token

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
                    response = requests.get(paginated_url, headers={"Authorization": f"Bearer {self.token}"})

                    # 处理token过期情况
                    if response.status_code == 401 or response.status_code == 403:
                        print(f"Token已过期或访问被拒绝，正在重新获取（第 {attempt + 1}/{retries} 次）...")
                        self.token = self.get_access_token()
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

                    break  # 成功获取数据，跳出重试循环

                except Exception as e:
                    print(f"检索数据时发生错误: {e}")
                    if attempt == retries - 1:  # 最后一次重试失败
                        print("达到最大重试次数，返回已获取的数据")
                        return search_results
                    print(f"2秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(2)

    def download_file(self, product_id: str, save_path: str):
        if os.path.exists(f"{save_path}.zip"):
            print(f"{save_path}.zip 已存在，跳过下载...")
            return
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        retries = 5  # 增加最大重试次数
        headers = {"Authorization": f"Bearer {self.token}"}
        chunk_size = 1024 * 1024 * 20  # 20MB

        for attempt in range(retries):
            try:
                # 先检查文件大小
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                if response.status_code == 401:
                    print(f"401 错误：尝试重新获取 Token（第 {attempt + 1}/{retries} 次）...")
                    self.token = self.get_access_token()
                    headers["Authorization"] = f"Bearer {self.token}"
                    continue

                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))

                # 创建进度条
                progress_bar = tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f"下载 {os.path.basename(save_path)}"
                )

                # 开始下载
                with open(f"{save_path}.tmp", "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                            downloaded += len(chunk)

                        # 检查是否下载完整
                        if downloaded >= total_size:
                            break

                progress_bar.close()

                # 检查文件完整性
                if os.path.getsize(f"{save_path}.tmp") == total_size:
                    os.rename(f"{save_path}.tmp", f"{save_path}.zip")
                    print(f"\n{save_path}.zip 下载完成！")
                    return
                else:
                    raise Exception("文件下载不完整，将重试")

            except Exception as e:
                print(f"\n下载 {save_path} 时发生错误: {e}")
                if os.path.exists(f"{save_path}.tmp"):
                    os.remove(f"{save_path}.tmp")
                if attempt < retries - 1:
                    print(f"等待5秒后进行第 {attempt + 2} 次重试...")
                    time.sleep(5)
                else:
                    print(f"下载 {save_path} 失败，已达到最大重试次数。")
                    return

    def download_all(self, search_url: str, save_folder: str, data_type: str, max_per_day: int):
        """
        下载所有检索结果，按照用户选择的类型筛选
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

        # 开始下载符合条件的文件
        for date, items in sorted(daily_results.items()):
            if len(items) > max_per_day:
                continue

            for item in items:
                product_id = item["Id"]
                file_name = item["Name"]
                save_path = os.path.join(save_folder, f"{file_name}_{date}")
                self.download_file(product_id, save_path)

            if items:  # 只有在有文件下载时才打印
                print(f"{date} 文件下载完毕。")

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
    save_folder = filedialog.askdirectory(title="选择保存目录")
    if not save_folder:
        messagebox.showerror("错误", "请选择保存目录！")
        return

    # 将输入的日期转换为 datetime 对象
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # 构造数据名称过滤条件
    # 将用户输入的名称拆分，过滤空白项
    data_names = [dn.strip() for dn in data_names_input.split(",") if dn.strip()]
    if not data_names:
        messagebox.showerror("错误", "请输入至少一个数据名称！")
        return

    # 根据每个数据名称构造条件，若数据名称为 "L2__NO2___" 则附加新的地理位置过滤条件
    name_conditions = []
    for dn in data_names:
        if dn == "L2__NO2___":
            condition = (
                f"(contains(Name,'{dn}') and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON ((-128.99 26.05, -63.54 26.05, "
                f"-63.54 52.81, -128.99 52.81, -128.99 26.05))'))"
            )
        else:
            condition = f"contains(Name,'{dn}')"
        name_conditions.append(condition)
    name_filter = "(" + " or ".join(name_conditions) + ")"

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
        downloader.download_all(search_url, save_folder, data_type, max_per_day)

        # 更新当前月的起始日期到下一个月
        current_start_date = current_end_date + timedelta(days=1)

# 主程序
if __name__ == "__main__":
    USER_NAME = "wanghaoyi@home.hpu.edu.cn"  # 替换为你的用户名
    PASSWORD = "Why23056510."  # 替换为你的密码

    root = tk.Tk()
    root.title("Sentinel 数据下载器")

    tk.Label(root, text="开始日期 (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=5)
    start_date_var = tk.StringVar(value="2023-01-01")
    tk.Entry(root, textvariable=start_date_var).grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="结束日期 (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5)
    end_date_var = tk.StringVar(value="2023-12-31")
    tk.Entry(root, textvariable=end_date_var).grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="数据名称:").grid(row=2, column=0, padx=10, pady=5)
    data_name_var = tk.StringVar(value="L2__NO2___")
    data_name_options = ["L1B_IR_SIR", "L1B_IR_UVN", "L1B_RA_BD1", "L1B_RA_BD2", "L1B_RA_BD3", "L1B_RA_BD4", "L1B_RA_BD5",
                         "L1B_RA_BD6", "L1B_RA_BD7", "L1B_RA_BD8", "L2__NO2___", "L2_HCHO___", "L2_CLOUD___", "L2_AER_AI", "L2_AER_LH"]
    ttk.Combobox(root, textvariable=data_name_var, values=data_name_options).grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="数据类型:").grid(row=3, column=0, padx=10, pady=5)
    data_type_var = tk.StringVar(value="全部")
    ttk.Combobox(root, textvariable=data_type_var, values=["NRTI", "RPRO", "OFFL", "全部"]).grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="每天最多下载数据数量:").grid(row=4, column=0, padx=10, pady=5)
    max_per_day_var = tk.StringVar(value="10")
    tk.Entry(root, textvariable=max_per_day_var).grid(row=4, column=1, padx=10, pady=5)

    tk.Button(root, text="开始下载", command=start_search).grid(row=5, column=0, columnspan=2, pady=20)

    root.mainloop()
