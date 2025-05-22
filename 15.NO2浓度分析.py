import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

# 设置全局 Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 加载数据
file_path = "D:\\DBN\\辐射值\\processed_result_with_ERA5.xlsx"
df = pd.read_excel(file_path)

# 确保 tropomi_time 为 datetime 格式
df['tropomi_time'] = pd.to_datetime(df['tropomi_time'])

# 添加时间相关列用于分析
df['year'] = df['tropomi_time'].dt.year
df['month'] = df['tropomi_time'].dt.month
df['date'] = df['tropomi_time'].dt.date

# 季节和类别的英文翻译
season_mapping = {'春季': 'Spring', '夏季': 'Summer', '秋季': 'Fall', '冬季': 'Winter'}
class_mapping = {
    '交通枢纽': 'Transportation Hub',
    '公共设施': 'Public Facility',
    '农业区': 'Agricultural Area',
    '商业区': 'Commercial Area',
    '居住区': 'Residential Area',
    '工业区': 'Industrial Area',
    '教育区': 'Educational Area',
    '未分类': 'Unclassified',
    '生态保护区': 'Ecological Protection Area'
}
day_type_mapping = {'工作日': 'Weekday', '星期天': 'Sunday'}

df['season_english'] = df['season'].map(season_mapping)
df['class_english'] = df['class'].map(class_mapping)
df['day_type_english'] = df['day_type'].map(day_type_mapping)

# 1. 空间分布分析
# 计算站点平均值
station_avg = df.groupby(['station_number', 'station_name', 'latitude', 'longitude'])[
    ['pgn_no2', 'strat_no2']].mean().reset_index()

# 创建空间分布图的函数（已修正）
def create_spatial_map(data, value_col, title, cmap='YlOrRd', scale_factor=1e6):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 缩放值以便于可视化
    values = data[value_col] * scale_factor

    # 移除异常值以优化颜色范围
    vmin = values.quantile(0.05)
    vmax = values.quantile(0.95)

    # 绘制散点图
    scatter = ax.scatter(
        data.longitude, data.latitude,
        c=values, cmap=cmap,
        s=80, alpha=0.8, edgecolor='k', linewidth=0.5,
        vmin=vmin, vmax=vmax
    )

    # 设置地图范围
    padding = 3  # 度数
    ax.set_extent([
        data.longitude.min() - padding,
        data.longitude.max() + padding,
        data.latitude.min() - padding,
        data.latitude.max() + padding
    ])

    # 添加颜色条（修正部分）
    pos = ax.get_position()  # 获取主轴位置
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # 手动创建颜色条轴
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(f'{value_col} (µmol/m²)')

    plt.title(title)
    # 注意：移除 tight_layout()，因为手动添加轴后可能导致重叠
    return fig, ax

# 创建空间分布图
fig1, ax1 = create_spatial_map(station_avg, 'pgn_no2', 'Spatial Distribution of PGN NO₂')
fig2, ax2 = create_spatial_map(station_avg, 'strat_no2', 'Spatial Distribution of Stratospheric NO₂', cmap='Blues')

# 2. 时间分析
# 计算每日平均值
daily_avg = df.groupby('date')[['pgn_no2', 'strat_no2']].mean().reset_index()

# 创建时间序列图
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(daily_avg['date'], daily_avg['pgn_no2'] * 1e6, label='PGN NO₂', color='darkred', linewidth=1.5)
ax3_twin = ax3.twinx()
ax3_twin.plot(daily_avg['date'], daily_avg['strat_no2'] * 1e6, label='Stratospheric NO₂', color='navy', linewidth=1.5)

# 格式化轴
ax3.set_xlabel('Date')
ax3.set_ylabel('PGN NO₂ (µmol/m²)', color='darkred')
ax3_twin.set_ylabel('Stratospheric NO₂ (µmol/m²)', color='navy')
ax3.tick_params(axis='y', colors='darkred')
ax3_twin.tick_params(axis='y', colors='navy')

# 格式化日期轴
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 添加网格和图例
ax3.grid(True, linestyle='--', alpha=0.6)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Time Series of PGN and Stratospheric NO₂ Concentrations')
plt.tight_layout()

# 3. 季节分析
# 计算季节平均值
seasonal_avg = df.groupby(['season', 'season_english'])[['pgn_no2', 'strat_no2']].mean().reset_index()
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
seasonal_avg['season_english'] = pd.Categorical(seasonal_avg['season_english'], categories=season_order, ordered=True)
seasonal_avg = seasonal_avg.sort_values('season_english')

# 创建季节对比图
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

# 按季节绘制 NO2 - 分开柱状图以便于可视化
ax4a.bar(seasonal_avg['season_english'], seasonal_avg['pgn_no2'] * 1e6, color='darkred', alpha=0.7, edgecolor='black')
ax4a.set_ylabel('PGN NO₂ (µmol/m²)')
ax4a.set_title('Seasonal Variation of PGN NO₂')
ax4a.grid(axis='y', linestyle='--', alpha=0.6)

ax4b.bar(seasonal_avg['season_english'], seasonal_avg['strat_no2'] * 1e6, color='navy', alpha=0.7, edgecolor='black')
ax4b.set_ylabel('Stratospheric NO₂ (µmol/m²)')
ax4b.set_title('Seasonal Variation of Stratospheric NO₂')
ax4b.grid(axis='y', linestyle='--', alpha=0.6)

# 添加数据标签
for ax in [ax4a, ax4b]:
    ax.set_xlabel('Season')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(ax.containers[0]):
        height = v.get_height()
        ax.text(i, height + 0.000002 * 1e6, f'{height:.2f}', ha='center')

plt.tight_layout()

# 4. 站点类别分析
# 计算类别平均值
class_avg = df.groupby(['class', 'class_english'])[['pgn_no2', 'strat_no2']].mean().reset_index()
class_avg = class_avg.sort_values('pgn_no2', ascending=False)

# 创建类别对比图
fig5, ax5 = plt.subplots(figsize=(14, 7))

x = np.arange(len(class_avg))
width = 0.35

rects1 = ax5.bar(x - width / 2, class_avg['pgn_no2'] * 1e6, width, label='PGN NO₂', color='darkred', alpha=0.7,
                 edgecolor='black')
rects2 = ax5.bar(x + width / 2, class_avg['strat_no2'] * 1e6, width, label='Stratospheric NO₂', color='navy', alpha=0.7,
                 edgecolor='black')

ax5.set_xlabel('Station Class')
ax5.set_ylabel('NO₂ Concentration (µmol/m²)')
ax5.set_title('NO₂ Concentration by Station Class')
ax5.set_xticks(x)
ax5.set_xticklabels(class_avg['class_english'], rotation=45, ha='right')
ax5.legend()
ax5.grid(axis='y', linestyle='--', alpha=0.6)

# 添加数据标签
for rect in rects1:
    height = rect.get_height()
    ax5.text(rect.get_x() + rect.get_width() / 2., height + 0.000001 * 1e6,
             f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()

# 5. 星期分析
# 将数字星期映射为名称
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
               4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_name'] = df['day_of_week'].map(day_mapping)

# 计算星期平均值
day_avg = df.groupby('day_name')[['pgn_no2', 'strat_no2']].mean().reset_index()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg['day_name'] = pd.Categorical(day_avg['day_name'], categories=day_order, ordered=True)
day_avg = day_avg.sort_values('day_name')

# 创建星期图
fig6, ax6 = plt.subplots(figsize=(12, 6))

x = np.arange(len(day_avg))
width = 0.35

rects1 = ax6.bar(x - width / 2, day_avg['pgn_no2'] * 1e6, width, label='PGN NO₂', color='darkred', alpha=0.7,
                 edgecolor='black')
rects2 = ax6.bar(x + width / 2, day_avg['strat_no2'] * 1e6, width, label='Stratospheric NO₂', color='navy', alpha=0.7,
                 edgecolor='black')

ax6.set_xlabel('Day of Week')
ax6.set_ylabel('NO₂ Concentration (µmol/m²)')
ax6.set_title('Day of Week Variations in NO₂ Concentrations')
ax6.set_xticks(x)
ax6.set_xticklabels(day_avg['day_name'])
ax6.legend()

# 高亮周末
ax6.axvspan(5 - 0.5, 6 + 0.5, color='lightgray', alpha=0.3, zorder=0)
ax6.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# 6. 与气象变量的相关性分析
# 选择相关 ERA5 变量
era5_cols = ['ERA5_t2m', 'ERA5_d2m', 'ERA5_u10', 'ERA5_v10', 'ERA5_blh', 'ERA5_tp']
era5_labels = {
    'ERA5_t2m': 'Temperature (2m)',
    'ERA5_d2m': 'Dew Point (2m)',
    'ERA5_u10': 'U Wind (10m)',
    'ERA5_v10': 'V Wind (10m)',
    'ERA5_blh': 'Boundary Layer Height',
    'ERA5_tp': 'Total Precipitation'
}
corr_cols = ['pgn_no2', 'strat_no2'] + era5_cols
no2_labels = {'pgn_no2': 'PGN NO₂', 'strat_no2': 'Stratospheric NO₂'}

# 计算相关性
corr_matrix = df[corr_cols].corr()

# 重命名列以便更好标注
corr_matrix_renamed = corr_matrix.rename(index={**no2_labels, **era5_labels},
                                         columns={**no2_labels, **era5_labels})

# 创建相关性热图
fig7, ax7 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix_renamed, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax7)
ax7.set_title('Correlation Between NO₂ and Meteorological Variables')
plt.tight_layout()

# 7. 季节空间分布
# 计算季节-空间平均值
seasonal_spatial = df.groupby(['season_english', 'station_number', 'latitude', 'longitude'])[
    ['pgn_no2']].mean().reset_index()

# 创建季节空间分布图
fig8, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for i, season in enumerate(season_order):
    ax = axes[i]
    season_data = seasonal_spatial[seasonal_spatial['season_english'] == season]

    # 添加地图特征
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 所有季节使用一致的颜色范围
    vmin = seasonal_spatial['pgn_no2'].quantile(0.05) * 1e6
    vmax = seasonal_spatial['pgn_no2'].quantile(0.95) * 1e6

    # 绘制数据
    scatter = ax.scatter(
        season_data.longitude, season_data.latitude,
        c=season_data['pgn_no2'] * 1e6, cmap='YlOrRd',
        s=70, alpha=0.8, edgecolor='k', linewidth=0.5,
        vmin=vmin, vmax=vmax
    )

    # 设置地图范围
    padding = 3  # 度数
    ax.set_extent([
        df.longitude.min() - padding,
        df.longitude.max() + padding,
        df.latitude.min() - padding,
        df.latitude.max() + padding
    ])

    ax.set_title(f'{season} PGN NO₂ Distribution')

# 添加颜色条
cbar_ax = fig8.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig8.colorbar(scatter, cax=cbar_ax)
cbar.set_label('PGN NO₂ (µmol/m²)')

plt.suptitle('Seasonal Spatial Distribution of PGN NO₂', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 1])

# 8. 综合洞察可视化
fig9, ((ax9a, ax9b), (ax9c, ax9d)) = plt.subplots(2, 2, figsize=(16, 14))

# 面板 1：按季节的 PGN NO2 与 Strat NO2 比率
ratio_by_season = seasonal_avg.copy()
ratio_by_season['ratio'] = ratio_by_season['pgn_no2'] / ratio_by_season['strat_no2']

ax9a.bar(ratio_by_season['season_english'], ratio_by_season['ratio'], color='purple', alpha=0.7, edgecolor='black')
ax9a.set_title('PGN/Stratospheric NO₂ Ratio by Season')
ax9a.set_xlabel('Season')
ax9a.set_ylabel('Ratio')
ax9a.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(ax9a.containers[0]):
    height = v.get_height()
    ax9a.text(i, height + 0.1, f'{height:.2f}', ha='center')

# 面板 2：PGN NO2 前五站点类别
top_classes = class_avg.head(5)
ax9b.bar(top_classes['class_english'], top_classes['pgn_no2'] * 1e6, color='darkred', alpha=0.7, edgecolor='black')
ax9b.set_title('Top 5 Station Classes by PGN NO₂')
ax9b.set_xlabel('Station Class')
ax9b.set_ylabel('PGN NO₂ (µmol/m²)')
ax9b.grid(axis='y', linestyle='--', alpha=0.6)
ax9b.set_xticklabels(top_classes['class_english'], rotation=45, ha='right')

# 面板 3：月度趋势
df['year_month'] = df['tropomi_time'].dt.to_period('M')
monthly_avg = df.groupby('year_month')[['pgn_no2', 'strat_no2']].mean().reset_index()
monthly_avg['date'] = monthly_avg['year_month'].dt.to_timestamp()

ax9c.plot(monthly_avg['date'], monthly_avg['pgn_no2'] * 1e6, 'o-', color='darkred', linewidth=2, label='PGN NO₂')
ax9c_twin = ax9c.twinx()
ax9c_twin.plot(monthly_avg['date'], monthly_avg['strat_no2'] * 1e6, 'o-', color='navy', linewidth=2,
               label='Stratospheric NO₂')

ax9c.set_title('Monthly Trends of NO₂ Concentrations')
ax9c.set_xlabel('Date')
ax9c.set_ylabel('PGN NO₂ (µmol/m²)', color='darkred')
ax9c_twin.set_ylabel('Stratospheric NO₂ (µmol/m²)', color='navy')
ax9c.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax9c.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax9c.xaxis.get_majorticklabels(), rotation=45, ha='right')
lines1, labels1 = ax9c.get_legend_handles_labels()
lines2, labels2 = ax9c_twin.get_legend_handles_labels()
ax9c.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax9c.grid(True, linestyle='--', alpha=0.6)

# 面板 4：PGN NO2 与关键气象变量的关系
key_era5 = 'ERA5_blh'  # 边界层高度通常有较强相关性
ax9d.scatter(df[key_era5], df['pgn_no2'] * 1e6, alpha=0.2, color='darkred')
ax9d.set_title(f'PGN NO₂ vs {era5_labels[key_era5]}')
ax9d.set_xlabel(era5_labels[key_era5])
ax9d.set_ylabel('PGN NO₂ (µmol/m²)')
ax9d.grid(True, linestyle='--', alpha=0.6)

# 添加趋势线
z = np.polyfit(df[key_era5], df['pgn_no2'] * 1e6, 1)
p = np.poly1d(z)
ax9d.plot(sorted(df[key_era5]), p(sorted(df[key_era5])), "r--", lw=2)
corr = np.corrcoef(df[key_era5], df['pgn_no2'])[0, 1]
ax9d.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax9d.transAxes,
          bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.suptitle('Comprehensive NO₂ Spatiotemporal Analysis Insights', fontsize=18, y=0.98)

# 显示所有图形
plt.show()