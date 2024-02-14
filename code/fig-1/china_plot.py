
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader as shpreader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.image import imread
from pylab import rcParams
import matplotlib
### Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 0.2,
          'font.size': 20,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 25,
          'xtick.direction': 'out',
          'ytick.labelsize': 25,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

shp_path = '/Users/weilab/Desktop/绘制中国地图/China_shp_cartopy/'
extent = [70, 140, 15, 60]  # 图3范围
# --- 加载中国矢量图信息
Chinese_land_territory_path = shp_path + 'China_land_territory/China_land_territory.shp'
Chinese_10dash_line_path = shp_path + 'China_10-dash_line/China_10-dash_line.shp'
world_countries_path = shp_path + 'world_countries/world_countries.shp'
city_path = shp_path + 'city/city.shp'
# --- 加载全球高分辨率地形
tif_path = "/Users/weilab/Desktop/绘制中国地图/"
NE_path = '/Users/weilab/Documents/My_work/PCSE/scripts/NEshp/NE.shp'
# shape_feature = ShapelyFeature(Reader(fname).geometries(),
#                                ccrs.PlateCarree(), facecolor='none')
# ax.add_feature(shape_feature)
# --- 加载站点信息
# df = pd.read_csv("Data1.csv")
# Read the Excel file
# df = pd.read_excel('data.xlsx')

# Print the first 5 rows of the DataFrame
# print(df.head())
prj = ccrs.PlateCarree()
fig = plt.figure(figsize=(12, 9), dpi=350)


def create_map():
    # 创建坐标系
    ax = fig.subplots(1, 1, subplot_kw={'projection': prj})
    ax.set_extent(extent, crs=prj)

    # 绘制中国陆地领土边界
    Chinese_land_territory = shpreader(Chinese_land_territory_path).geometries()
    Chinese_land_territory = cfeat.ShapelyFeature(Chinese_land_territory,
                                                  prj, edgecolor='gray',alpha=0.7,
                                                  facecolor='none')
    ax.add_feature(Chinese_land_territory, linewidth=1)

    # # 绘制city边界
    # citys = shpreader(city_path).geometries()
    # citys = cfeat.ShapelyFeature(citys,
    #                              prj, edgecolor='k', alpha=0.3,
    #                              facecolor='none')
    # ax.add_feature(world_countries, linewidth=1.5)
    dongbei = shpreader(NE_path).geometries()
    dongbei = cfeat.ShapelyFeature(dongbei,
                                 prj, edgecolor='k',
                                 facecolor='none')
    ax.add_feature(dongbei, linewidth=1.5)

    # 绘制中国十段线
    Chinese_10dash_line = shpreader(Chinese_10dash_line_path).geometries()
    Chinese_10dash_line = cfeat.ShapelyFeature(Chinese_10dash_line,
                                               prj, edgecolor='r')
    ax.add_feature(Chinese_10dash_line, linewidth=2)

    # 绘制世界各国领土边界
    world_countries = shpreader(world_countries_path).geometries()
    world_countries = cfeat.ShapelyFeature(world_countries,
                                           prj, edgecolor='k', alpha=0.3,
                                           facecolor='none')
    ax.add_feature(world_countries, linewidth=0.5)

    # —— 加载地形数据
    # — 加载低分辨率地形
    # ax.stock_img()
    # --加载高分辨率地形
    ax.imshow(
        imread(tif_path + 'NE1_50M_SR_W.tif'),
        origin='upper', transform=prj,
        extent=[-180, 180, -90, 90]
    )

    # 绘制网格点
    gl = ax.gridlines(crs=prj, draw_labels=True, linewidth=1.2, color='k',
                      alpha=0.5, linestyle='--')
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 10, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 10, 10))
    ax.tick_params(axis='x',  pad=50)
    # -- 加载详细地理信息
    # --加载分辨率为50的海岸线
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # --加载分辨率为50的河流~
    # ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
    # --加载分辨率为50的湖泊
    ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)

    # --设置南海子图

    left, bottom, width, height = 0.698, 0.19, 0.23, 0.27
    ax2 = fig.add_axes(
        [left, bottom, width, height],
        projection=prj
    )
    ax2.add_feature(Chinese_land_territory, linewidth=0.6, zorder=2)
    # 绘制中国十段线
    Chinese_10dash_line = shpreader(Chinese_10dash_line_path).geometries()
    Chinese_10dash_line = cfeat.ShapelyFeature(Chinese_10dash_line,
                                               prj, edgecolor='r')
    ax2.add_feature(Chinese_10dash_line, linewidth=2)
    ax2.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)  # 加载分辨率为50的海岸线
    ax2.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)  # 加载分辨率为50的河流
    ax2.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)  # 加载分辨率为50的湖泊
    ax2.set_extent([105, 125, 0, 25])
    ax2.stock_img()
    # —— 加载地形数据
    # — 加载低分辨率地形
    # ax.stock_img()
    # --加载高分辨率地形
    ax2.add_feature(Chinese_land_territory, linewidth=1)
    ax2.imshow(
        imread(tif_path + 'NE1_50M_SR_W.tif'),
        origin='upper', transform=prj,
        extent=[-180, 180, -90, 90]
    )
    lon0, lat0, length = 80, 23, 1000
    # 左下角经度，左下角纬度，比例尺长度
    ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length / 111, colors="black", ls="-", lw=2, label='%d km' % (length))
    ax.vlines(x=lon0, ymin=lat0 - 0.45, ymax=lat0 + 0.45, colors="black", ls="-", lw=2)
    ax.vlines(x=lon0 + length / 111, ymin=lat0 - 0.45, ymax=lat0 + 0.45, colors="black", ls="-", lw=2)
    ax.text(lon0 + length / 2 / 111, lat0 + 0.45, f'{length} km', horizontalalignment='center')
    return ax


# --调用刚才定义的地图函数
ax = create_map()
# ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.3, color='gray', alpha=0.5)

# df['lon'] = df['lon'].astype(np.float64)
# df['lat'] = df['lat'].astype(np.float64)
#
# # --绘制散点图
# ax.scatter(
#     df['lon'].values,
#     df['lat'].values,
#     marker='o',
#     s=10,
#     color="blue"
# )

# --添加浮标名称
# for i, j, k in list(zip(df['lon'].values, df['lat'].values, df['city_eng'].values)):
#     ax.text(i - 0.8, j + 0.2, k, fontsize=4)

# --添加标题&设置字号
# title = f'distribution of city'
# ax.set_title(title, fontsize=18)

# plt.show()
fig.savefig('demo.png')
# fig.savefig('demo.eps')
# from pyecharts.charts import Map
# from pyecharts import options as opts
#
# # 示例数据
# provinces = ["黑龙江", "吉林", "辽宁"]
# values = [100, 200, 300]
#
# # 创建 Map 实例
# map_chart = (
#     Map()
#     .add("东北三省", [list(z) for z in zip(provinces, values)], "china")
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="中国地图"),
#         visualmap_opts=opts.VisualMapOpts(max_=max(values), is_piecewise=True),
#     )
# )
#
# # 将东北三省设置为特定颜色
# map_chart.set_series_opts(label_opts=opts.LabelOpts(is_show=False),
#                           itemstyle_opts={"normal": {"color": "#ff4500"}})
#
# # 生成 HTML 文件
# map_chart.render("china_map_with_dongbei.png")
#
#
# from pyecharts import options as opts
# from pyecharts.charts import Map
# import random
#
# ultraman = [
#     ['黑龙江', 10],
#     ['吉林', 10],
#     ['辽宁', 10]
# ]
# # 设置怪兽存在的相关省份，并设置初始数量为0
# monster = [
#     ['黑龙江', 10],
#     ['吉林', 10],
#     ['辽宁', 10]
# ]
#
#
# def data_filling(array):
#     '''
#      作用：给数组数据填充随机数
#     '''
#     for i in array:
#         # 随机生成1到1000的随机数
#         i[1] = random.randint(1,1000)#10
#         print(i)
#
#
# data_filling(ultraman)
# data_filling(monster)
#
# def create_china_map():
#     '''
#      作用：生成中国地图
#     '''
#     (
#         Map()
#         .add(
#             series_name="",
#             data_pair=ultraman,
#             maptype="china",
#             is_map_symbol_show=True
#         )
#
#         # 设置标题
#         .set_global_opts(title_opts=opts.TitleOpts(title=""),
#                          visualmap_opts=opts.VisualMapOpts(max_=10, is_piecewise=True))
#         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#
#         # 生成本地html文件
#         .render("中国地图.html")
#     )
# def create_china_map():
#     '''
#      作用：生成中国地图
#     '''
#     (
#         Map()
#         .add(
#             series_name="奥特曼",
#             data_pair=ultraman,
#             maptype="china",
#             # 是否默认选中
#             is_selected=True
#         )
#         .add(
#             series_name="怪兽",
#             data_pair=monster,
#             maptype="china",
#         )
#         .set_global_opts(
#         # 设置标题
#         title_opts=opts.TitleOpts(title="中国地图"),
#         # 设置分段显示
#         visualmap_opts=opts.VisualMapOpts(max_=1000, is_piecewise=True)
#         )
#         # 生成本地html文件
#         .render("中国地图.html")
#     )


# create_china_map()
