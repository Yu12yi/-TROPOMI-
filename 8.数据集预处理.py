import pandas as pd
import numpy as np
import holidays
from geopy.geocoders import Nominatim, ArcGIS
from geopy.distance import geodesic
from collections import Counter
import jieba
import re
from typing import Optional, List, Dict
import time
import logging  # 导入 logging 模块

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加自定义词典
custom_dict = {
    # 学校相关
    "小学", "中学", "大学", "学院", "校区", "campus",
    # 商业区相关
    "商圈", "商务区", "市中心", "downtown", "plaza", "mall",
    # 居住区相关
    "社区", "小区", "住宅区", "公寓", "apartment", "residence",
    # 工业相关
    "工业园", "产业园", "科技园", "工厂", "industrial park", "factory",
    # 农业相关
    "农场", "果园", "农业区", "farm", "plantation",
    # 交通相关
    "机场", "港口", "车站", "highway", "airport", "station",
    # 其他
    "公园", "医院", "体育场", "park", "hospital", "stadium"
}

for word in custom_dict:
    jieba.add_word(word)

class AddressAnalyzer:
    """
    地址分析器，用于清理地址、提取关键词、分析地址类型，并进行邻域类型校验。
    """
    def __init__(self):
        """
        初始化地址分析器，设置地理编码器和缓存。
        """
        self.nominatim = Nominatim(user_agent="address_analyzer_v4", timeout=10) # 修改 user_agent 版本
        self.arcgis = ArcGIS(timeout=10)
        self.cache = {}
        self.search_radius_km = 7.5  # 设置搜索半径为7.5km
        # 定义详细的区域类型关键词
        self.area_types = {
    '教育区': {
        'cn': [
            '大学', '学院', '学校', '校区', '图书馆', '研究所', '幼儿园', '托儿所',
            '中小学', '职校', '技校', '培训中心', '教学楼', '实验楼', '操场', '教育机构',
            '学术', '科研', '教研', '学区', '书店', '教育', '学院路', '大学城', '智谷',
            '创新中心', '高教区', '留学生公寓', '教师公寓', '学生宿舍'
        ],
        'en': [
            'university', 'college', 'school', 'campus', 'library', 'institute',
            'kindergarten', 'preschool', 'elementary school', 'middle school',
            'high school', 'vocational school', 'technical school', 'training center',
            'teaching building', 'laboratory building', 'playground', 'educational institution',
            'academic', 'research', 'scholarly', 'education district', 'bookstore',
            'education', 'college road', 'university town', 'innovation park',
            'higher education zone', 'international student apartment', 'faculty apartment',
            'student dormitory', 'research institute'
        ]
    },
    '商业区': {
        'cn': [
            '商圈', '商业', '市中心', '购物', '商场', '广场', '写字楼', '办公', 'CBD',
            '金融街', '步行街', '商店', '店铺', '超市', '酒店', '餐饮', '娱乐', '会展中心',
            '商务中心', '贸易', '金融', '投资', '企业', '公司', '大厦', '商贸', '商业街',
            '购物中心', '步行街区', '金融区', '经济开发区', '总部基地', '创业园', '孵化器',
            '商业综合体', '商业步行街', '购物公园', '奥特莱斯', '商业广场', '商务楼宇', '写字间'
        ],
        'en': [
            'downtown', 'plaza', 'mall', 'shopping', 'business', 'office', 'commercial',
            'CBD', 'financial district', 'pedestrian street', 'store', 'shop', 'supermarket',
            'hotel', 'restaurant', 'entertainment', 'convention center', 'business center',
            'trade', 'finance', 'investment', 'enterprise', 'corporation', 'building',
            'commercial trade', 'commercial street', 'shopping center', 'pedestrian zone',
            'financial area', 'economic development zone', 'headquarters base',
            'innovation park', 'incubator', 'commercial complex', 'commercial pedestrian street',
            'shopping park', 'outlets', 'commercial plaza', 'office building', 'office space'
        ]
    },
    '居住区': {
        'cn': [
            '社区', '小区', '住宅', '公寓', '家园', '花园', '别墅', '居民区', '宿舍',
            '住宅小区', '住宅楼', '民宅', '家属院', '生活区', '居委会', '物业', '安居房',
            '经济适用房', '廉租房', '公租房', '回迁房', '安置房', '商品房', '居民楼', '家属区',
            '生活小区', '住宅区', '别墅区', '公寓楼', '花园小区', '家园小区', '新村', '里弄',
            '坊', '苑', '邸', '府', '庭', '轩', '榭', '居', '舍', '寓', '邸', '第', '村', '庄', '屯'
        ],
        'en': [
            'community', 'residential', 'apartment', 'housing', 'garden', 'estate',
            'villa', 'resident area', 'dormitory', 'residential area', 'residential building',
            'private residence', 'family compound', 'living area', 'residents committee',
            'property management', 'affordable housing', 'economical housing',
            'low-rent housing', 'public rental housing', 'relocation housing',
            'resettlement housing', 'commodity housing', 'residential building',
            'family area', 'living community', 'residential zone', 'villa district',
            'apartment building', 'garden community', 'homeland community', 'new village',
            'lane', 'neighborhood', 'garden', 'mansion', 'residence', 'court', 'house',
            'home', 'apartment', 'mansion', 'No.', 'village', 'manor', 'settlement'
        ]
    },
    '工业区': {
        'cn': [
            '工业园', '产业园', '科技园', '工厂', '制造', '加工', '开发区', '加工厂',
            '车间', '仓库', '物流园', '生产基地', '出口加工区', '高新区', '产业基地',
            '重工业', '轻工业', '制造业', '工业基地', '厂房', '工业区', '产业区', '科技园区',
            '物流中心', '仓储中心', '生产车间', '装配车间', '研发中心', '测试中心', '质检中心',
            '工业大道', '产业大道', '科技大道', '园区', '基地', '中心', '车间', '厂', '库',
            '物流', '制造', '加工', '生产', '研发', '测试', '质检', '出口', '进口', '保税区',
            '出口加工', '高新技术产业', '战略新兴产业', '先进制造业', '智能制造', '绿色制造'
        ],
        'en': [
            'industrial', 'factory', 'manufacturing', 'plant', 'technology park',
            'development zone', 'processing plant', 'workshop', 'warehouse',
            'logistics park', 'production base', 'export processing zone',
            'high-tech zone', 'industrial base', 'heavy industry', 'light industry',
            'manufacturing industry', 'industrial zone', 'factory building',
            'industrial area', 'industrial zone', 'technology park area',
            'logistics center', 'warehousing center', 'production workshop',
            'assembly workshop', 'R&D center', 'testing center', 'quality inspection center',
            'industrial avenue', 'industrial road', 'technology avenue', 'park', 'base',
            'center', 'workshop', 'factory', 'warehouse', 'logistics', 'manufacturing',
            'processing', 'production', 'R&D', 'testing', 'quality inspection', 'export',
            'import', 'bonded zone', 'export processing', 'high-tech industry',
            'strategic emerging industry', 'advanced manufacturing', 'intelligent manufacturing',
            'green manufacturing', 'industrial park'
        ]
    },
    '农业区': {
        'cn': [
            '农场', '果园', '农业', '种植', '养殖', '田园', '农村', '乡村', '耕地',
            '农田', '菜地', '茶园', '牧场', '渔场', '林场', '养殖场', '灌溉', '农产品',
            '生态农业', '农业合作社', '农家乐', '采摘园', '农业科技园', '农业示范园',
            '农贸市场', '农资市场', '种子站', '农技站', '水稻田', '旱地', '梯田', '山地',
            '丘陵', '平原', '盆地', '草原', '牧区', '林区', '渔区', '垦区', '农垦', '农林',
            '田', '地', '园', '场', '庄园', '农庄', '生态园', '科技园', '示范园', '合作社',
            '农户', '农家', '田间', '地头', '垄上', '埂边', '乡间小路', '田埂', '垄', '埂',
            '乡村公路', '农用', '农业机械', '农药', '化肥', '种子', '农膜', '农具', '农舍',
            '农家院', '粮仓', '谷仓', '粮田', '菜田', '棉田', '油菜田', '果树', '蔬菜', '粮食',
            '棉花', '油料', '瓜果', '茶叶', '药材', '花卉', '苗木', '禽', '畜', '鱼', '虾',
            '蟹', '贝', '藻', '菌', '粮', '棉', '油', '果', '菜', '茶', '药', '花', '苗',
            '禽', '畜', '鱼', '虾', '蟹', '贝', '藻', '菌', '农', '牧', '渔', '林', '垦'
        ],
        'en': [
            'farm', 'plantation', 'agricultural', 'ranch', 'orchard', 'rural area',
            'countryside', 'farmland', 'agricultural field', 'vegetable garden',
            'tea garden', 'pasture', 'fishery', 'forest farm', 'breeding farm',
            'irrigation', 'agricultural products', 'ecological agriculture',
            'agricultural cooperative', 'agritainment', 'picking garden',
            'agricultural science and technology park', 'agricultural demonstration park',
            'farmers market', 'agricultural materials market', 'seed station',
            'agricultural technology station', 'paddy field', 'dry land', 'terraced field',
            'mountainous area', 'hills', 'plain', 'basin', 'grassland', 'pastoral area',
            'forestry area', 'fishing area', 'reclamation area', 'state farm',
            'agriculture and forestry', 'field', 'land', 'garden', 'farm', 'manor',
            'farmstead', 'ecological park', 'science and technology park',
            'demonstration park', 'cooperative', 'farmer', 'farmhouse', 'field',
            'field head', 'ridge', 'ditch', 'country road', 'field ridge', 'ridge',
            'ditch', 'rural highway', 'agricultural use', 'agricultural machinery',
            'pesticide', 'fertilizer', 'seed', 'agricultural film', 'farm tool',
            'farmhouse', 'farmyard', 'granary', 'barn', 'grain field', 'vegetable field',
            'cotton field', 'rapeseed field', 'fruit tree', 'vegetable', 'grain', 'cotton',
            'oil crop', 'melon and fruit', 'tea', 'medicinal material', 'flower', 'seedling',
            'poultry', 'livestock', 'fish', 'shrimp', 'crab', 'shellfish', 'algae',
            'bacteria', 'grain', 'cotton', 'oil', 'fruit', 'vegetable', 'tea', 'medicine',
            'flower', 'seedling', 'poultry', 'livestock', 'fish', 'shrimp', 'crab',
            'shellfish', 'algae', 'bacteria', 'agriculture', 'animal husbandry', 'fishery',
            'forestry', 'reclamation', 'agricultural area'
        ]
    },
    '交通枢纽': {
        'cn': [
            '机场', '港口', '车站', '高速', '铁路', '地铁', '航站楼', '候机楼', '货运港',
            '客运站', '公交站', '轻轨', '码头', '立交桥', '收费站', '交通中心', '物流中心',
            '航运', '客运', '货运', '交通运输', '火车站', '汽车站', '客运码头', '货运码头',
            '机场航站楼', '地铁站', '轻轨站', '公交枢纽', '长途汽车站', '公交总站', '轨道交通',
            '城际铁路', '高速公路', '国道', '省道', '县道', '乡道', '城市快速路', '环线',
            '高架桥', '隧道', '桥梁', '立交', '枢纽', '中心站', '总站', '客运中心', '货运中心',
            '交通', '运输', '航', '港', '站', '路', '线', '高速入口', '高速出口', '匝道',
            '服务区', '停车区', '加油站', '充电站', '换乘中心', '集散中心', '物流园区', '保税物流'
        ],
        'en': [
            'airport', 'port', 'station', 'highway', 'railway', 'metro', 'terminal building',
            'airport terminal', 'cargo port', 'passenger station', 'bus station', 'light rail',
            'wharf', 'overpass', 'toll station', 'transportation center', 'logistics center',
            'shipping', 'passenger transport', 'freight transport', 'transportation',
            'railway station', 'bus station', 'passenger terminal', 'cargo terminal',
            'airport terminal', 'metro station', 'light rail station', 'bus hub',
            'long-distance bus station', 'bus terminus', 'rail transit', 'intercity railway',
            'expressway', 'national highway', 'provincial highway', 'county road',
            'township road', 'urban expressway', 'ring road', 'overpass bridge', 'tunnel',
            'bridge', 'interchange', 'hub', 'central station', 'terminus station',
            'passenger transport center', 'freight transport center', 'transportation',
            'transport', 'aviation', 'port', 'station', 'road', 'line', 'highway entrance',
            'highway exit', 'ramp', 'service area', 'parking area', 'gas station',
            'charging station', 'transfer center', 'distribution center', 'logistics park',
            'bonded logistics', 'transport hub'
        ]
    },
    '公共设施': {
        'cn': [
            '公园', '医院', '体育场', '政府', '博物馆', '广场', '图书馆', '文化馆',
            '展览馆', '纪念馆', '法院', '派出所', '消防局', '邮局', '银行', '剧院',
            '电影院', '健身房', '游泳馆', '公共服务', '市政', '行政', '文化', '医疗',
            '教育', '娱乐', '休闲', '市民中心', '行政中心', '文化中心', '医疗中心',
            '教育中心', '体育中心', '活动中心', '服务中心', '办事处', '管理局', '委员会',
            '中心广场', '文化广场', '体育广场', '休闲广场', '公园绿地', '公共绿地',
            '市政公园', '城市公园', '社区公园', '专科医院', '综合医院', '社区医院',
            '门诊部', '急诊中心', '体检中心', '疾控中心', '卫生中心', '体育馆', '运动场',
            '体育中心', '健身中心', '游泳池', '公共泳池', '政府机关', '行政机关', '事业单位',
            '党政机关', '法院', '检察院', '公安局', '派出所', '消防队', '消防站', '邮政局',
            '邮政支局', '银行', '银行分行', '银行支行', '剧院', '电影院', '影剧院', '文化宫',
            '少年宫', '老年活动中心', '青少年活动中心', '妇女儿童活动中心', '残疾人活动中心',
            '社区服务中心', '市民服务中心', '行政服务中心', '公共服务平台', '政务服务中心',
            '便民服务中心', '咨询中心', '信息中心', '展览中心', '展示中心', '会议中心',
            '活动中心', '服务中心', '办事大厅', '服务大厅', '接待中心', '咨询台', '服务台',
            '公共厕所', '公共电话亭', '公共自行车站点', '公共汽车站', '地铁站', '轻轨站',
            '公交站', '出租车站点', '停车场', '公共停车场', '收费停车场', '免费停车场',
            '公共交通', '市政设施', '公共设施', '公用设施', '市政工程', '公共工程',
            '公益设施', '福利设施', '文化设施', '体育设施', '医疗设施', '教育设施',
            '娱乐设施', '休闲设施', '服务设施', '办事设施', '咨询设施', '信息设施',
            '展览设施', '展示设施', '会议设施', '活动设施', '接待设施', '停车设施',
            '交通设施', '环卫设施', '消防设施', '邮政设施', '银行设施', '剧院设施',
            '电影院设施', '健身房设施', '游泳馆设施', '公共服务设施', '市政公共设施',
            '公共事业', '市政建设', '公共建设', '公益事业', '福利事业', '文化事业',
            '体育事业', '医疗事业', '教育事业', '娱乐事业', '休闲事业', '服务事业',
            '办事事业', '咨询事业', '信息事业', '展览事业', '展示事业', '会议事业',
            '活动事业', '接待事业', '停车事业', '交通事业', '环卫事业', '消防事业',
            '邮政事业', '银行业', '剧院业', '电影院业', '健身房业', '游泳馆业',
            '公共服务业', '市政公共事业', '公共事业管理', '市政建设管理', '公共建设管理',
            '公益事业管理', '福利事业管理', '文化事业管理', '体育事业管理', '医疗事业管理',
            '教育事业管理', '娱乐事业管理', '休闲事业管理', '服务事业管理', '办事事业管理',
            '咨询事业管理', '信息事业管理', '展览事业管理', '展示事业管理', '会议事业管理',
            '活动事业管理', '接待事业管理', '停车事业管理', '交通事业管理', '环卫事业管理',
            '消防事业管理', '邮政事业管理', '银行管理', '剧院管理', '电影院管理', '健身房管理',
            '游泳馆管理', '公共服务管理', '市政公共管理', '公共管理', '市政管理', '公益管理',
            '福利管理', '文化管理', '体育管理', '医疗管理', '教育管理', '娱乐管理', '休闲管理',
            '服务管理', '办事管理', '咨询管理', '信息管理', '展览管理', '展示管理', '会议管理',
            '活动管理', '接待管理', '停车管理', '交通管理', '环卫管理', '消防管理', '邮政管理',
            '银行管理', '剧院管理', '电影院管理', '健身房管理', '游泳馆管理', '公共服务中心',
            '市政公共中心', '公共中心', '市政中心', '公益中心', '福利中心', '文化中心',
            '体育中心', '医疗中心', '教育中心', '娱乐中心', '休闲中心', '服务中心',
            '办事中心', '咨询中心', '信息中心', '展览中心', '展示中心', '会议中心',
            '活动中心', '接待中心', '停车中心', '交通中心', '环卫中心', '消防中心',
            '邮政中心', '银行中心', '剧院中心', '电影院中心', '健身房中心', '游泳馆中心',
            'public facilities'
        ],
        'en': [
            'park', 'hospital', 'stadium', 'government', 'museum', 'square', 'library',
            'cultural center', 'exhibition hall', 'memorial hall', 'court', 'police station',
            'fire station', 'post office', 'bank', 'theater', 'cinema', 'gym',
            'swimming pool', 'public service', 'municipal', 'administrative', 'cultural',
            'medical', 'educational', 'entertainment', 'leisure', 'civic center',
            'administrative center', 'cultural center', 'medical center', 'education center',
            'sports center', 'activity center', 'service center', 'office', 'administration',
            'committee', 'central square', 'cultural square', 'sports square',
            'leisure square', 'park green space', 'public green space', 'municipal park',
            'city park', 'community park', 'specialized hospital', 'general hospital',
            'community hospital', 'outpatient department', 'emergency center',
            'physical examination center', 'disease control center', 'health center',
            'gymnasium', 'sports field', 'sports center', 'fitness center', 'swimming pool',
            'public swimming pool', 'government agencies', 'administrative agencies',
            'public institutions', 'party and government agencies', 'court',
            'procuratorate', 'public security bureau', 'police station', 'fire brigade',
            'fire station', 'post office', 'post office branch', 'bank', 'bank branch',
            'bank sub-branch', 'theater', 'cinema', 'cinema theater', 'cultural palace',
            'children\'s palace', 'senior citizens activity center',
            'youth activity center', 'women and children activity center',
            'disabled persons activity center', 'community service center',
            'citizen service center', 'administrative service center',
            'public service platform', 'government affairs service center',
            'public convenience service center', 'consultation center', 'information center',
            'exhibition center', 'display center', 'conference center', 'activity center',
            'service center', 'service hall', 'reception center', 'consultation desk',
            'service desk', 'public toilet', 'public telephone booth',
            'public bicycle station', 'bus stop', 'metro station', 'light rail station',
            'bus station', 'taxi stand', 'parking lot', 'public parking lot',
            'paid parking lot', 'free parking lot', 'public transportation',
            'municipal facilities', 'public facilities', 'public utilities',
            'municipal engineering', 'public works', 'public welfare facilities',
            'welfare facilities', 'cultural facilities', 'sports facilities',
            'medical facilities', 'educational facilities', 'entertainment facilities',
            'leisure facilities', 'service facilities', 'office facilities',
            'consultation facilities', 'information facilities', 'exhibition facilities',
            'display facilities', 'conference facilities', 'activity facilities',
            'reception facilities', 'parking facilities', 'transportation facilities',
            'environmental sanitation facilities', 'fire protection facilities',
            'postal facilities', 'banking facilities', 'theater facilities',
            'cinema facilities', 'gym facilities', 'swimming pool facilities',
            'public service facilities', 'municipal public facilities',
            'public utilities', 'municipal construction', 'public construction',
            'public welfare undertakings', 'welfare undertakings', 'cultural undertakings',
            'sports undertakings', 'medical undertakings', 'educational undertakings',
            'entertainment undertakings', 'leisure undertakings', 'service undertakings',
            'office undertakings', 'consultation undertakings', 'information undertakings',
            'exhibition undertakings', 'display undertakings', 'conference undertakings',
            'activity undertakings', 'reception undertakings', 'parking undertakings',
            'transportation undertakings', 'environmental sanitation undertakings',
            'fire protection undertakings', 'postal undertakings', 'banking industry',
            'theater industry', 'cinema industry', 'gym industry', 'swimming pool industry',
            'public service industry', 'municipal public utilities industry',
            'public utilities management', 'municipal construction management',
            'public construction management', 'public welfare undertakings management',
            'welfare undertakings management', 'cultural undertakings management',
            'sports undertakings management', 'medical undertakings management',
            'educational undertakings management', 'entertainment undertakings management',
            'leisure undertakings management', 'service undertakings management',
            'office undertakings management', 'consultation undertakings management',
            'information undertakings management', 'exhibition undertakings management',
            'display undertakings management', 'conference undertakings management',
            'activity undertakings management', 'reception undertakings management',
            'parking undertakings management', 'transportation undertakings management',
            'environmental sanitation undertakings management',
            'fire protection undertakings management', 'postal undertakings management',
            'banking management', 'theater management', 'cinema management',
            'gym management', 'swimming pool management', 'public service management',
            'municipal public management', 'public management', 'municipal management',
            'public welfare management', 'welfare management', 'cultural management',
            'sports management', 'medical management', 'educational management',
            'entertainment management', 'leisure management', 'service management',
            'office management', 'consultation management', 'information management',
            'exhibition management', 'display management', 'conference management',
            'activity management', 'reception management', 'parking management',
            'transportation management', 'environmental sanitation management',
            'fire protection management', 'postal management', 'bank management',
            'theater management', 'cinema management', 'gym management',
            'swimming pool management', 'public service center', 'municipal public center',
            'public center', 'municipal center', 'public welfare center', 'welfare center',
            'cultural center', 'sports center', 'medical center', 'education center',
            'entertainment center', 'leisure center', 'service center', 'office center',
            'consultation center', 'information center', 'exhibition center',
            'display center', 'conference center', 'activity center', 'reception center',
            'parking center', 'transportation center', 'environmental sanitation center',
            'fire protection center', 'postal center', 'bank center', 'theater center',
            'cinema center', 'gym center', 'swimming pool center', 'public facilities'
        ]
    },
    '生态保护区': {
        'cn': [
            '自然保护区', '森林公园', '湿地公园', '地质公园', '海洋公园', '风景名胜区', '生态公园', '国家公园',
            '省级自然保护区', '市级自然保护区', '郊野公园', '野生动物保护区', '植物园', 'arboretum', '生态 заповед区',
            '生物多样性保护区', '水源保护区', '风景林', '保护性耕地', '重要湿地', '国际重要湿地', '鸟类保护区', '鱼类保护区',
            '珍稀植物保护区', '珍稀动物保护区', '基因库', '生态恢复区', '绿色廊道', '生态屏障', '生物走廊', '生态敏感区',
            '环境脆弱区', '自然景观保护区', '文化景观保护区', '遗产地', '世界遗产地', '自然遗产地', '文化和自然双重遗产地', '湿地',
            '草原', '森林', '山地', '丘陵', '河流', '湖泊', '海岸线', '海岛', '峡谷', '溶洞', '沙漠', '冰川',
            '火山', '地热区', '生物栖息地', '迁徙通道', '候鸟栖息地', '鱼类洄游通道', '珊瑚礁', '红树林', '海草床', '河口',
            '三角洲', '滩涂', '沼泽', '泥炭地', '高山草甸', '苔原', '原始森林', '次生林', '人工林', '防护林',
            '经济林', '特种用途林', '种子园', '母树林', '基因库', '生态农业示范区', '有机农业示范区', '绿色食品基地',
            '无公害农产品基地', '休闲农业区', '生态旅游区', '环境教育基地', '科普教育基地', '环保宣传基地', '绿色学校', '生态社区',
            '可持续发展示范区', '低碳示范区', '循环经济示范区', '生态文明示范区', '美丽乡村示范区', '绿色城镇示范区',
            '环境友好型产业园区', '生态工业园区', '循环经济产业园区'
        ],
        'en': [
            'nature reserve', 'forest park', 'wetland park', 'geopark', 'marine park', 'scenic area', 'ecological park', 'national park',
            'provincial nature reserve', 'municipal nature reserve', 'country park', 'wildlife sanctuary', 'botanical garden', 'arboretum', 'ecological reserve',
            'biodiversity conservation area', 'water source protection area', 'scenic forest', 'protected cultivated land', 'important wetland', 'wetland of international importance', 'bird sanctuary', 'fish sanctuary',
            'rare plant protection area', 'rare animal protection area', 'gene bank', 'ecological restoration area', 'green corridor', 'ecological barrier', 'biological corridor', 'ecologically sensitive area',
            'environmentally vulnerable area', 'natural landscape conservation area', 'cultural landscape conservation area', 'heritage site', 'world heritage site', 'natural heritage site', 'mixed natural and cultural heritage site', 'wetland',
            'grassland', 'forest', 'mountain area', 'hilly area', 'river', 'lake', 'coastline', 'island', 'canyon', 'cave', 'desert', 'glacier',
            'volcano', 'geothermal area', 'biological habitat', 'migration corridor', 'migratory bird habitat', 'fish migration channel', 'coral reef', 'mangrove forest', 'seagrass bed', 'estuary',
            'delta', 'tidal flat', 'marsh', 'peatland', 'alpine meadow', 'tundra', 'virgin forest', 'secondary forest', 'artificial forest', 'protection forest',
            'economic forest', 'special purpose forest', 'seed orchard', 'mother tree forest', 'gene pool', 'ecological agriculture demonstration area', 'organic agriculture demonstration area', 'green food base',
            'pollution-free agricultural product base', 'leisure agriculture area', 'ecotourism area', 'environmental education base', 'science popularization education base', 'environmental protection publicity base', 'green school', 'ecological community',
            'sustainable development demonstration area', 'low-carbon demonstration area', 'circular economy demonstration area', 'ecological civilization demonstration area', 'beautiful countryside demonstration area', 'green town demonstration area',
            'environmentally friendly industrial park', 'ecological industrial park', 'circular economy industrial park'
        ]
    },
    '医疗健康区': {
        'cn': [
            '综合医院', '专科医院', '社区医院', '中医院', '妇幼保健院', '儿童医院', '传染病医院', '精神病医院',
            '肿瘤医院', '康复医院', '护理院', '疗养院', '卫生院', '卫生所', '诊所', '医务室', '急救中心', '血液中心',
            '疾控中心', '体检中心', '健康管理中心', '医疗中心', '康复中心', '护理中心', '养老中心', '临终关怀中心',
            '卫生服务中心', '社区卫生服务中心', '乡镇卫生院', '村卫生室', '医院门诊部', '医院住院部', '特诊部', '专家门诊',
            '特色专科', '重点学科', '临床医学中心', '医学研究中心', '实验医学中心', '远程医疗中心', '医学影像中心', '病理诊断中心',
            '基因检测中心', '健康大数据中心', '医药研发中心', '医疗器械研发中心', '医疗产业园', '生物医药产业园', '健康产业园', '养老产业园',
            '康复产业园', '医疗器械生产基地', '药品生产基地', '疫苗生产基地', '血液制品生产基地', '医疗物流中心', '医用耗材配送中心', '医疗废物处理中心',
            '医疗培训中心', '医学教育中心', '护理培训中心', '药学教育中心', '医学院', '护理学院', '药学院', '卫生学校',
            '医科大学', '医药大学', '护理大学', '康复大学', '健康管理学院', '公共卫生学院', '医疗器械公司', '医药公司',
            '保健品公司', '养老服务公司', '康复服务公司', '健康咨询公司', '医疗保险公司', '医药电商平台', '医疗健康app开发区', '互联网医院',
            '云药房', '智慧医疗示范区', '远程诊疗中心', '家庭医生工作室', '健康小屋', '养老驿站', '社区 care station', '健康步道',
            '健身广场', '康复公园', '医疗街', '药店街', '保健品专卖街', '养老服务街', '康复器材街', '医美机构聚集区',
            '月子中心聚集区', '基因检测服务中心', '干细胞治疗中心', '抗衰老中心', '辅助生殖中心', '心理咨询中心', '营养咨询中心', '运动康复中心',
            '职业病防治中心', '罕见病诊疗中心', '慢病管理中心', '老年病防治中心', '儿科医疗中心', '妇产科医疗中心', '心血管病医疗中心', '肿瘤诊疗中心',
            '神经系统疾病诊疗中心', '呼吸系统疾病诊疗中心', '消化系统疾病诊疗中心', '内分泌系统疾病诊疗中心', '免疫系统疾病诊疗中心', '骨科医疗中心', '眼科医疗中心', '口腔医疗中心',
            '皮肤科医疗中心', '精神心理科医疗中心', '急诊创伤中心', '重症医学中心', '检验医学中心', '病理诊断中心', '医学影像中心', '超声诊断中心',
            '内镜诊疗中心', '介入治疗中心', '放射治疗中心', '物理治疗中心', '职业康复中心', '心理康复中心', '社区康复中心', '居家康复中心',
            '特色疗法中心', '名医工作室', '专家诊疗中心', '会诊中心', '远程会诊中心', '医学图书馆', '医学博物馆', '医学档案馆'
        ],
        'en': [
            'general hospital', 'specialized hospital', 'community hospital', 'traditional chinese medicine hospital', 'maternal and child health hospital', 'children\'s hospital', 'infectious disease hospital', 'psychiatric hospital',
            'cancer hospital', 'rehabilitation hospital', 'nursing home', 'sanatorium', 'health center', 'clinic', 'medical office', 'first aid center', 'blood center', 'disease control center',
            'physical examination center', 'health management center', 'medical center', 'rehabilitation center', 'nursing center', 'elderly care center', 'hospice care center', 'health service center',
            'community health service center', 'township health center', 'village clinic', 'hospital outpatient department', 'hospital inpatient department', 'special clinic department', 'expert outpatient clinic', 'featured specialty',
            'key discipline', 'clinical medical center', 'medical research center', 'experimental medical center', 'telemedicine center', 'medical imaging center', 'pathological diagnosis center', 'genetic testing center',
            'health big data center', 'pharmaceutical research and development center', 'medical device research and development center', 'medical industrial park', 'biomedicine industrial park', 'health industrial park', 'elderly care industrial park', 'rehabilitation industrial park',
            'medical device production base', 'drug production base', 'vaccine production base', 'blood product production base', 'medical logistics center', 'medical consumables distribution center', 'medical waste treatment center', 'medical training center',
            'medical education center', 'nursing training center', 'pharmaceutical education center', 'medical college', 'nursing college', 'pharmacy college', 'health school', 'medical university',
            'pharmaceutical university', 'nursing university', 'rehabilitation university', 'health management college', 'public health college', 'medical device company', 'pharmaceutical company', 'health product company',
            'elderly care service company', 'rehabilitation service company', 'health consulting company', 'medical insurance company', 'pharmaceutical e-commerce platform', 'medical and health app development zone', 'internet hospital', 'cloud pharmacy',
            'smart healthcare demonstration zone', 'remote diagnosis and treatment center', 'family doctor studio', 'health kiosk', 'elderly care station', 'community care station', 'health trail', 'fitness square',
            'rehabilitation park', 'medical street', 'pharmacy street', 'health product specialty street', 'elderly care service street', 'rehabilitation equipment street', 'medical beauty institution cluster area', 'postpartum care center cluster area',
            'genetic testing service center', 'stem cell therapy center', 'anti-aging center', 'assisted reproduction center', 'psychological counseling center', 'nutritional counseling center', 'sports rehabilitation center', 'occupational disease prevention and treatment center',
            'rare disease diagnosis and treatment center', 'chronic disease management center', 'geriatric disease prevention and treatment center', 'pediatric medical center', 'obstetrics and gynecology medical center', 'cardiovascular disease medical center', 'tumor diagnosis and treatment center', 'nervous system disease diagnosis and treatment center',
            'respiratory system disease diagnosis and treatment center', 'digestive system disease diagnosis and treatment center', 'endocrine system disease diagnosis and treatment center', 'immune system disease diagnosis and treatment center', 'orthopedic medical center', 'ophthalmology medical center', 'stomatology medical center', 'dermatology medical center',
            'psychiatric medical center', 'emergency trauma center', 'critical care medicine center', 'laboratory medicine center', 'pathology diagnosis center', 'medical imaging center', 'ultrasound diagnosis center', 'endoscopy diagnosis and treatment center',
            'interventional therapy center', 'radiotherapy center', 'physical therapy center', 'occupational rehabilitation center', 'psychological rehabilitation center', 'community rehabilitation center', 'home rehabilitation center', 'featured therapy center',
            'famous doctor studio', 'expert diagnosis and treatment center', 'consultation center', 'remote consultation center', 'medical library', 'medical museum', 'medical archives'
        ]
    },
    '军事区': {
        'cn': [
            '军事基地', '军事禁区', '军营', '军区', '战区', '海军基地', '空军基地', '火箭军基地',
            '战略支援部队基地', '军事院校', '国防大学', '军事科学院', '军事博物馆', '军港', '军用机场', '军用铁路',
            '军用公路', '军用仓库', '军械库', '弹药库', '油料库', '军事指挥中心', '作战指挥中心', '训练基地',
            '演习场', '靶场', '试验场', '科研基地', '军工企业', '军品生产基地', '军需物资仓库', '军事装备维修中心',
            '军事医院', '军事疗养院', '军事干休所', '军事家属区', '军事管理区', '军事管制区', '边防站', '哨所',
            '雷达站', '导弹阵地', '防空阵地', '军事通信枢纽', '军事交通枢纽', '军事物流中心', '军事后勤保障基地', '军事战略要地',
            '国防前沿', '海防前线', '陆地边境', '边境口岸', '边境检查站', '军事演习区', '实弹演习区', '禁航区',
            '禁飞区', '军事演练场', '阅兵场', '军事展览馆', '军事装备展示区', '国防教育基地', '爱国主义教育基地', '军事主题公园',
            '军事体验营地', '军事文化街区', '军人服务社', '军人俱乐部', '军人接待站', '军人公寓', '军人家属招待所', '退役军人服务中心',
            '军转干部培训中心', '军事研究机构', '国防科技实验室', '军事工程研究院', '军事医学研究所', '军事装备研究所', '军事战略研究中心',
            '军事政策研究中心', '军事历史研究中心', '军事理论研究中心', '军事法研究中心', '军事伦理研究中心', '军事心理研究中心', '军事社会科学研究中心', '军事自然科学研究中心',
            '军事技术科学研究中心', '军事系统工程研究中心', '军事运筹学研究中心', '军事信息科学研究中心', '军事指挥与控制研究中心', '军事战略评估中心', '国防经济研究中心', '军事后勤研究中心',
            '军事装备采购中心', '军事人力资源研究中心', '军事教育训练研究中心', '军事政策与战略咨询中心', '军事国际合作研究中心', '军事危机管理研究中心', '军事安全战略研究中心', '军事战略文化研究中心',
            '军事高技术研究中心', '军事新兴领域研究中心', '军事网络空间安全研究中心', '军事人工智能研究中心', '军事智能化研究中心', '军事无人系统研究中心', '军事空间作战研究中心', '军事网络空间作战研究中心'
        ],
        'en': [
            'military base', 'military restricted zone', 'military camp', 'military region', 'war zone', 'naval base', 'air force base', 'rocket force base',
            'strategic support force base', 'military academy', 'national defence university', 'academy of military science', 'military museum', 'military port', 'military airport', 'military railway',
            'military highway', 'military warehouse', 'armory', 'ammunition depot', 'fuel depot', 'military command center', 'combat command center', 'training base',
            'exercise field', 'firing range', 'testing ground', 'scientific research base', 'military industrial enterprise', 'military product production base', 'military supplies warehouse', 'military equipment maintenance center',
            'military hospital', 'military sanatorium', 'military cadre sanatorium', 'military family area', 'military management area', 'military control area', 'border post', 'sentry post',
            'radar station', 'missile position', 'air defence position', 'military communication hub', 'military transportation hub', 'military logistics center', 'military logistics support base', 'military strategic location',
            'national defence frontier', 'sea defence frontline', 'land border', 'border port', 'border inspection station', 'military exercise area', 'live fire exercise area', 'no-navigation zone',
            'no-fly zone', 'military training ground', 'parade ground', 'military exhibition hall', 'military equipment display area', 'national defence education base', 'patriotism education base', 'military theme park',
            'military experience camp', 'military culture block', 'military servicemen\'s club', 'military club', 'military reception station', 'military apartment', 'military family guest house', 'veterans service center',
            'military transfer cadre training center', 'military research institution', 'national defence science and technology laboratory', 'military engineering research institute', 'military medical research institute', 'military equipment research institute', 'military strategy research center',
            'military policy research center', 'military history research center', 'military theory research center', 'military law research center', 'military ethics research center', 'military psychology research center', 'military social science research center', 'military natural science research center',
            'military technical science research center', 'military system engineering research center', 'military operations research center', 'military information science research center', 'military command and control research center', 'military strategic assessment center', 'national defence economic research center', 'military logistics research center',
            'military equipment procurement center', 'military human resources research center', 'military education and training research center', 'military policy and strategy consulting center', 'military international cooperation research center', 'military crisis management research center', 'military security strategy research center', 'military strategic culture research center',
            'military high technology research center', 'military emerging fields research center', 'military cyberspace security research center', 'military artificial intelligence research center', 'military intelligentization research center', 'military unmanned system research center', 'military space operations research center', 'military cyberspace operations research center'
        ]
    },
    '文旅区': {
        'cn': [
            '旅游景区', '旅游度假区', '旅游景点', '文化景区', '历史文化街区', '风情小镇', '特色小镇', '旅游小镇',
            '度假村', '酒店集群', '民宿集群', '温泉度假村', '滑雪度假村', '海滨度假村', '山地度假村', '乡村旅游区',
            '农业旅游区', '工业旅游区', '生态旅游区', '红色旅游区', '古镇', '古村落', '文物保护单位', '历史建筑群',
            '博物馆群', '美术馆群', '艺术馆群', '文化馆群', '纪念馆群', '科技馆群', '规划馆', '档案馆',
            '图书馆群', '剧院群', '影院群', '演艺中心', '文化中心', '艺术中心', '创意产业园', '文化产业园',
            '动漫产业园', '影视产业园', '演艺产业园', '艺术品交易中心', '文创产品交易中心', '旅游商品交易中心', '特色商品街', '小吃街',
            '美食街', '酒吧街', '咖啡馆街', '茶馆街', '民俗文化村', '民族文化村', '传统文化街区', '非物质文化遗产传承基地',
            '传统工艺美术街区', '文学艺术创作基地', '影视拍摄基地', '演艺演出场所', '艺术展览馆', '文物商店', '古玩市场', '书画市场',
            '工艺品市场', '旅游纪念品商店', '特色产品店', '旅游信息咨询中心', '游客服务中心', '旅游集散中心', '旅游交通枢纽', '旅游停车场',
            '旅游厕所', '旅游步道', '观景台', '露营地', '自驾车营地', '房车营地', '游船码头', '游艇码头',
            '航空旅游基地', '热气球基地', '滑翔伞基地', '攀岩基地', '徒步线路', '自行车道', '旅游专线', '景区直通车',
            '旅游班车', '旅游观光巴士', '旅游服务公司', '旅行社', '酒店管理公司', '景区管理公司', '旅游开发公司', '文化传媒公司',
            '演艺公司', '艺术品拍卖公司', '文创产品设计公司', '旅游规划设计院', '景区策划公司', '旅游咨询机构', '文化交流中心', '艺术交流中心',
            '国际文化节', '旅游节庆活动', '民俗节庆活动', '文化演出活动', '艺术展览活动', '创意市集', '文创集市', '旅游商品展销会',
            '特色美食节', '啤酒节', '音乐节', '艺术节', '电影节', '戏剧节', '舞蹈节', '动漫节',
            '书展', '画展', '雕塑展', '摄影展', '艺术装置展', '文化创意设计大赛', '旅游商品创意大赛', '美食烹饪大赛',
            '摄影大赛', '微电影大赛', '剧本创作大赛', '舞蹈比赛', '音乐比赛', '艺术表演比赛', '文化创意创业园', '旅游创新创业孵化器'
        ],
        'en': [
            'tourist scenic area', 'tourist resort area', 'tourist attraction', 'cultural scenic area', 'historic and cultural block', 'charming town', 'characteristic town', 'tourist town',
            'resort village', 'hotel cluster', 'bed and breakfast cluster', 'hot spring resort', 'ski resort', 'seaside resort', 'mountain resort', 'rural tourism area',
            'agricultural tourism area', 'industrial tourism area', 'ecotourism area', 'red tourism area', 'ancient town', 'ancient village', 'cultural relics protection unit', 'historic building complex',
            'museum cluster', 'art gallery cluster', 'art museum cluster', 'cultural center cluster', 'memorial hall cluster', 'science and technology museum cluster', 'planning exhibition hall', 'archives hall',
            'library cluster', 'theater cluster', 'cinema cluster', 'performing arts center', 'cultural center', 'art center', 'creative industry park', 'cultural industry park',
            'animation industry park', 'film and television industry park', 'performing arts industry park', 'artworks trading center', 'cultural and creative products trading center', 'tourist products trading center', 'characteristic commodity street', 'snack street',
            'food street', 'bar street', 'cafe street', 'tea house street', 'folk culture village', 'ethnic culture village', 'traditional culture block', 'intangible cultural heritage inheritance base',
            'traditional arts and crafts block', 'literature and art creation base', 'film and television shooting base', 'performing arts venue', 'art exhibition hall', 'cultural relics shop', 'antique market', 'calligraphy and painting market',
            'arts and crafts market', 'tourist souvenir shop', 'featured product shop', 'tourist information and consultation center', 'tourist service center', 'tourist distribution center', 'tourist transportation hub', 'tourist parking lot',
            'tourist toilet', 'tourist trail', 'observation deck', 'campsite', 'self-driving camp', 'rv camp', 'cruise ship terminal', 'yacht marina',
            'aviation tourism base', 'hot air balloon base', 'paragliding base', 'rock climbing base', 'hiking route', 'bicycle lane', 'tourist special line', 'scenic spot direct bus',
            'tourist shuttle bus', 'tourist sightseeing bus', 'tourist service company', 'travel agency', 'hotel management company', 'scenic area management company', 'tourism development company', 'cultural media company',
            'performing arts company', 'art auction company', 'cultural and creative product design company', 'tourism planning and design institute', 'scenic area planning company', 'tourist consulting agency', 'cultural exchange center', 'art exchange center',
            'international culture festival', 'tourism festival activities', 'folk festival activities', 'cultural performance activities', 'art exhibition activities', 'creative bazaar', 'cultural and creative fair', 'tourist commodity fair',
            'featured food festival', 'beer festival', 'music festival', 'arts festival', 'film festival', 'drama festival', 'dance festival', 'animation festival',
            'book fair', 'painting exhibition', 'sculpture exhibition', 'photography exhibition', 'art installation exhibition', 'cultural and creative design competition', 'tourist commodity creative competition', 'gourmet cooking competition',
            'photography contest', 'micro film competition', 'scriptwriting competition', 'dance competition', 'music competition', 'art performance competition', 'cultural and creative entrepreneurship park', 'tourism innovation and entrepreneurship incubator'
        ]
    },
    '郊区': {
        'cn': [
            '住宅区', '别墅区', '花园小区', '低密度住宅区', '郊外', '近郊', '远郊', '卫星城',
            '城乡结合部', '绿化带', '公园绿地', '郊区公园', '社区中心', '购物中心', '学校', '幼儿园',
            '体育场', '运动场', '停车场', '公交车站', '轻轨站', '地铁站', '高速公路入口', '主要干道',
            '环城公路', '产业园区', '科教园区', '高新技术开发区', '物流园区', '仓储区', '农业观光园'
        ],
        'en': [
            'Residential area', 'Suburb', 'Suburban area', 'Outskirts', 'Suburbia', 'Commuter town', 'Bedroom community', 'Green belt',
            'Park land', 'Suburban park', 'Community center', 'Shopping mall', 'School', 'Kindergarten', 'Stadium', 'Sports field',
            'Parking lot', 'Bus stop', 'Light rail station', 'Subway station', 'Highway entrance', 'Main road', 'Ring road', 'Industrial park',
            'Science park', 'Tech park', 'Logistics park', 'Warehousing district', 'Agricultural park'
        ]
    },
    '市区': {
        'cn': [
            '商业区', '中央商务区', '金融区', '购物街', '步行街', '商业步行街', '购物中心', '百货商店',
            '超市', '写字楼', '办公楼', '公寓楼', '高层住宅', '酒店', '饭店', '餐厅',
            '咖啡馆', '剧院', '电影院', '博物馆', '美术馆', '展览馆', '图书馆', '大学',
            '高中', '小学', '地铁站', '火车站', '长途汽车站', '公交枢纽', '出租车站点', '公共交通',
            '公园', '广场', '市政府', '政府机关', '法院', '警察局', '医院', '社区服务中心',
            '文化宫', '青少年宫', '体育馆', '游泳馆', '停车场', '老城区', '历史街区', '居民区',
            '高档住宅区'
        ],
        'en': [
            'Central business district (CBD)', 'Downtown', 'City center', 'Urban area', 'Commercial district', 'Financial district', 'Shopping street', 'Pedestrian street',
            'Shopping mall', 'Department store', 'Supermarket', 'Office building', 'Apartment building', 'High-rise residential', 'Hotel', 'Restaurant',
            'Cafe', 'Theater', 'Cinema', 'Museum', 'Art gallery', 'Exhibition hall', 'Library', 'University',
            'High school', 'Primary school', 'Subway station', 'Train station', 'Bus station', 'Bus terminal', 'Taxi stand', 'Public transportation',
            'City park', 'Square', 'City hall', 'Government building', 'Court house', 'Police station', 'Hospital', 'Community center',
            'Cultural center', 'Youth center', 'Gymnasium', 'Swimming pool', 'Parking garage', 'Old town', 'Historic district', 'Residential district',
            'Upscale residential area'
        ]
    },
   '科教研发区': {
        'cn': [
            '科研院所', '高等院校', '重点实验室', '工程中心', '技术中心', '研发中心', '实验基地', '中试基地',
            '创新中心', '创业中心', '孵化器', '加速器', '产业技术研究院', '科学中心', '技术转移中心', '科技服务中心',
            '知识产权服务中心', '技术交易市场', '科技成果转化基地', '产学研合作基地', '院士工作站', '博士后科研工作站', '海归创业园', '留学生创业园',
            '大学生创业园', '科技企业孵化器', '高新技术创业服务中心', '科技金融服务中心', '技术咨询服务中心', '科技信息服务中心', '科技中介服务机构', '科技成果评估机构',
            '科技成果推广机构', '科技人才服务机构', '科技培训机构', '科技传播机构', '科技普及基地', '科普场馆', '科技博物馆', '科技展览馆',
            '科技体验馆', '科学公园', '科技主题公园', '创新街区', '智力密集区', '人才特区', '知识经济示范区', '创新型城市示范区',
            '智慧城市示范区', '低碳技术示范区', '绿色技术示范区', '循环经济技术示范区', '生态技术示范区', '先进制造技术示范区', '智能制造技术示范区', '生物技术示范区',
            '信息技术示范区', '新材料技术示范区', '新能源技术示范区', '节能环保技术示范区', '农业高新技术示范区', '现代农业科技示范区', '工业互联网示范区', '大数据产业示范区',
            '人工智能产业示范区', '集成电路产业示范区', '生物医药产业示范区', '新能源汽车产业示范区', '新材料产业示范区', '航空航天产业示范区', '海洋工程装备产业示范区', '智能装备产业示范区',
            '机器人产业示范区', '新一代信息技术产业示范区', '战略性新兴产业示范区', '未来产业先导区', '原始创新策源地', '关键核心技术攻关基地', '重大科技基础设施', '大科学装置',
            '科学大数据中心', '超级计算中心', '云计算中心', '智能计算中心', '区块链技术创新中心', '量子信息技术创新中心', '类脑智能技术创新中心', '未来网络技术创新中心',
            '前沿交叉科学研究中心', '交叉学科研究平台', '协同创新中心', '国际科技合作基地', '国际联合实验室', '国际科技组织总部', '国际学术交流中心', '高端人才公寓',
            '专家公寓', '青年科学家公寓', '博士后公寓', '留学生公寓', '创新咖啡', '创业沙龙', '技术论坛', '学术会议',
            '科技展览会', '创新创业大赛', '科技成果发布会', '技术需求对接会', '知识产权交易会', '技术转移对接会', '科技人才招聘会', '科技成果路演中心',
            '技术转移服务站', '知识产权服务站', '科技金融服务站', '科技咨询服务站', '科技信息服务站', '科技中介服务站'
        ],
        'en': [
            'scientific research institute', 'institutions of higher education', 'key laboratory', 'engineering center', 'technology center', 'research and development center', 'experimental base', 'pilot scale test base',
            'innovation center', 'entrepreneurship center', 'incubator', 'accelerator', 'industrial technology research institute', 'science center', 'technology transfer center', 'science and technology service center',
            'intellectual property service center', 'technology trading market', 'science and technology achievement transformation base', 'industry-university-research cooperation base', 'academician workstation', 'postdoctoral research workstation', 'overseas returnees entrepreneurship park', 'international students entrepreneurship park',
            'university students entrepreneurship park', 'science and technology business incubator', 'high-tech business incubation service center', 'science and technology financial service center', 'technology consulting service center', 'science and technology information service center', 'science and technology intermediary service organization', 'science and technology achievement evaluation agency',
            'science and technology achievement promotion agency', 'science and technology talent service agency', 'science and technology training institution', 'science and technology communication agency', 'science and technology popularization base', 'science popularization venue', 'science and technology museum', 'science and technology exhibition hall',
            'science and technology experience hall', 'science park', 'science and technology theme park', 'innovation block', 'intellectually intensive area', 'talent special zone', 'knowledge economy demonstration zone', 'innovative city demonstration zone',
            'smart city demonstration zone', 'low-carbon technology demonstration zone', 'green technology demonstration zone', 'circular economy technology demonstration zone', 'ecological technology demonstration zone', 'advanced manufacturing technology demonstration zone', 'intelligent manufacturing technology demonstration zone', 'biotechnology demonstration zone',
            'information technology demonstration zone', 'new material technology demonstration zone', 'new energy technology demonstration zone', 'energy saving and environmental protection technology demonstration zone', 'agricultural high-tech demonstration zone', 'modern agricultural science and technology demonstration zone', 'industrial internet demonstration zone', 'big data industry demonstration zone',
            'artificial intelligence industry demonstration zone', 'integrated circuit industry demonstration zone', 'biomedicine industry demonstration zone', 'new energy vehicle industry demonstration zone', 'new material industry demonstration zone', 'aerospace industry demonstration zone', 'marine engineering equipment industry demonstration zone', 'intelligent equipment industry demonstration zone',
            'robot industry demonstration zone', 'new generation information technology demonstration zone', 'strategic emerging industry demonstration zone', 'future industry pioneer zone', 'original innovation source', 'key core technology tackling base', 'major science and technology infrastructure', 'big science device',
            'scientific big data center', 'supercomputing center', 'cloud computing center', 'intelligent computing center', 'blockchain technology innovation center', 'quantum information technology innovation center', 'brain-inspired intelligence technology innovation center', 'future network technology innovation center',
            'frontier interdisciplinary science research center', 'interdisciplinary research platform', 'collaborative innovation center', 'international science and technology cooperation base', 'international joint laboratory', 'headquarters of international science and technology organizations', 'international academic exchange center', 'high-end talent apartment',
            'expert apartment', 'young scientist apartment', 'postdoctoral apartment', 'international student apartment', 'innovation coffee', 'entrepreneurship salon', 'technology forum', 'academic conference',
            'science and technology exhibition', 'innovation and entrepreneurship competition', 'science and technology achievement release conference', 'technology demand matching meeting', 'intellectual property fair', 'technology transfer matching meeting', 'science and technology talent recruitment fair', 'science and technology achievement roadshow center',
            'technology transfer service station', 'intellectual property service station', 'science and technology financial service station', 'science and technology consulting service station', 'science and technology information service station', 'science and technology intermediary service station'
        ]
    },
    '重工业区': { # 新增类别：美国常见NO2排放区 - 重工业区
        'cn': [
            '炼油厂', '化工厂', '钢铁厂', '发电厂(燃煤)', '制造综合体(重型)', '冶炼厂', '工厂区(重型)', '工业园区(重型)',
            '重型制造区', '石化工厂', '石油精炼厂', '燃煤发电厂',  '金属加工厂', '装配线(重型)', '汽车制造厂', '造船厂',
            '铸造厂', '水泥厂', '纸浆和造纸厂', '矿区(工业)', '采掘场(工业)'
        ],
        'en': [
            'refinery', 'chemical plant', 'steel mill', 'power plant coal', 'manufacturing complex heavy', 'smelter', 'factory district heavy', 'industrial park heavy',
            'heavy manufacturing area', 'petrochemical plant', 'oil refinery', 'coal power plant',  'metal processing plant', 'assembly line heavy', 'automotive plant', 'shipyard',
            'foundry', 'cement plant', 'pulp and paper mill', 'mining area industrial', 'extraction site industrial'
        ]
    },
    '港口工业区': { # 新增类别：美国常见NO2排放区 - 港口工业区
        'cn': [
            '港口', '海港', '码头', '造船厂', '货运码头', '集装箱港口', '海运码头', '海军港口',
            '渡轮码头', '港务局', '海滨(工业)', '航运通道', '海事工业区', '沿海工业区', '港区', '港湾区',
            '码头区',  '货运港', '海港', '内陆港', '物流枢纽(港口)', '配送中心(港口)', '码头区域', '海运工业园',
            '入境口岸', '船舶装卸区', '卸货码头', '港口设施', '干船坞', '海军造船厂', '商业港口', '工业海滨'
        ],
        'en': [
            'port', 'harbor', 'dock', 'shipyard', 'cargo terminal', 'container port', 'marine terminal', 'naval port',
            'ferry terminal', 'port authority', 'waterfront industrial', 'shipping channel', 'maritime industrial area', 'coastal industrial zone', 'port district', 'harbor area',
            'docklands',  'freight port', 'ocean port', 'inland port', 'logistics hub port', 'distribution center port', 'wharf area', 'marine industry park',
            'port of entry', 'ship loading area', 'unloading docks', 'port facilities', 'dry dock', 'naval shipyard', 'commercial port', 'industrial waterfront'
        ]
    },
    '机场区域': { # 新增类别：美国常见NO2排放区 - 机场区域
        'cn': [
            '机场', '航空港', '飞机场', '空军基地', '飞行枢纽', '国际机场', '区域机场', '通用航空机场',
            '机场航站楼', '跑道', '控制塔', '机库', '停机坪', '滑行道', '飞机维修基地', '货运机场',
            '航空工业园', '机场工业区', '飞行航线', '空中交通管制中心', '机场管理局', '机场区', '航空航天工业区', '航空学院',
            '飞行员培训中心', '航空博物馆', '机场酒店区', '汽车租赁中心(机场)', '机场停车场', '机场巴士', '机场高速公路', '机场铁路连接线'
        ],
        'en': [
            'airport', 'airfield', 'aerodrome', 'air base', 'flight hub', 'international airport', 'regional airport', 'general aviation airport',
            'airport terminal', 'runway', 'control tower', 'hangar', 'apron', 'taxiway', 'aircraft maintenance base', 'cargo airport',
            'aviation industrial park', 'airport industrial area', 'flight path', 'air traffic control center', 'airport authority', 'airport district', 'aerospace industry zone', 'aviation academy',
            'pilot training center', 'aviation museum', 'airport hotel zone', 'rental car center airport', 'airport parking', 'airport shuttle', 'airport expressway', 'airport rail link'
        ]
    },
    '高密度交通区': { # 新增类别：美国常见NO2排放区 - 高密度交通区
        'cn': [
            '高速公路交汇处', '高速公路立交桥', '主干道', '交通走廊', '拥堵区域', '高峰时段区', '交通堵塞区', '交通繁忙街道',
            '卡车路线', '公交线路(密集)', '商业干道', '主街(繁忙)', '市中心核心区(交通)', '市中心(交通)', '城市网格(交通)', '主干道',
            '支路', '高速公路出口(城市)', '高速公路入口(城市)', '环城公路(内环)', '环路(内环)', '高架桥(主要)', '隧道(主要)', '桥梁(主要)',
            '十字路口(繁忙)', '环岛(繁忙)',  '交通环岛(繁忙)', '交通枢纽(道路)', '公交总站(城市)', '卡车停靠站(城市)', '停车场区', '送货区(繁忙)'
        ],
        'en': [
            'highway junction', 'freeway interchange', 'major roadway', 'traffic corridor', 'congested area', 'rush hour zone', 'gridlock zone', 'heavy traffic street',
            'truck route', 'bus route dense', 'commercial thoroughfare', 'main street busy', 'downtown core traffic', 'city center traffic', 'urban grid traffic', 'arterial road',
            'collector road', 'expressway exit urban', 'highway entrance urban', 'beltway inner', 'ring road inner', 'overpass major', 'tunnel major', 'bridge major',
            'intersection busy', 'roundabout busy',  'traffic circle busy', 'transit hub road', 'bus terminal city', 'truck stop city', 'parking garage district', 'delivery zone heavy'
        ]
    },
    '能源生产区': { # 新增类别：美国常见NO2排放区 - 能源生产区
        'cn': [
            '发电厂', '炼油厂', '发电站', '能源厂', '燃料加工厂', '天然气厂', '油田', '气田',
            '炼油联合体', '能源生产设施', '发电厂址', '能源工业园', '燃料库', '石油储存设施', '天然气储存设施', '输电线路走廊',
            '管道走廊', '能源基础设施区',  '钻井 site', '采掘场(能源)', '矿区(能源)', '地热发电厂', '风力发电场(工业)', '太阳能发电场(工业)',
            '生物质发电厂', '水力发电坝(工业)', '核电站', '化石燃料发电厂', '可再生能源区(工业)', '能源研究中心(工业)', '能源科技园'
        ],
        'en': [
            'power plant', 'refinery', 'generating station', 'energy plant', 'fuel processing plant', 'natural gas plant', 'oil field', 'gas field',
            'refining complex', 'energy production facility', 'power generation site', 'energy industrial park', 'fuel depot', 'oil storage facility', 'gas storage facility', 'transmission line corridor',
            'pipeline corridor', 'energy infrastructure zone',   'drilling site', 'extraction site energy', 'mining area energy', 'geothermal plant', 'wind farm industrial', 'solar farm industrial',
            'biomass power plant', 'hydroelectric dam industrial', 'nuclear power plant', 'fossil fuel power plant', 'renewable energy zone industrial', 'energy research center industrial', 'energy technology park'
        ]
    }
}

    def get_address(self, latitude: float, longitude: float) -> Optional[str]:
        """
        根据经纬度获取地址，使用缓存和 Nominatim/ArcGIS 地理编码服务。
        """
        coordinates = (latitude, longitude)

        if coordinates in self.cache:
            return self.cache[coordinates]

        for service_name, geolocator in [('Nominatim', self.nominatim), ('ArcGIS', self.arcgis)]:
            try:
                # 尝试使用较大范围搜索地址
                location = None
                if service_name == 'Nominatim':
                    # Nominatim支持搜索半径
                    location = geolocator.reverse(coordinates, language='zh-CN', exactly_one=True, zoom=15)
                else:
                    # ArcGIS没有直接的半径参数，但可以通过其他方式调整搜索精度
                    location = geolocator.reverse(coordinates)
            
                if location and location.address:
                    self.cache[coordinates] = location.address
                    logging.info(f"使用 {service_name} 获取到地址: {location.address} for {coordinates}")
                    return location.address
            except Exception as e:
                logging.error(f"Error using {service_name} for coordinates {coordinates}: {e}")
                continue  # 尝试下一个地理编码服务

        return None

    def clean_address(self, address: str) -> str:
        """
        清理和标准化地址文本，移除特殊字符和多余空格。
        """
        address = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', address)
        address = re.sub(r'\s+', ' ', address).strip()
        return address

    def extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取中英文关键词。
        """
        chinese_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', text))
        english_text = ' '.join(re.findall(r'[a-zA-Z]+', text))

        chinese_words = list(jieba.cut(chinese_text))
        english_words = english_text.lower().split()

        return chinese_words + english_words

    def analyze_keywords(self, addresses: list) -> Dict[str, Dict[str, int]]:
        """
        分析地址列表中的关键词频率和地址类型分布。（此函数在分类功能中不再直接使用，但保留以供后续分析）
        """
        keyword_stats = {category: {'cn': [], 'en': []} for category in self.area_types.keys()}
        address_types = []

        for address in addresses:
            if not address:
                continue

            clean_addr = self.clean_address(address)
            words = self.extract_keywords(clean_addr)

            address_type = self.classify_address(clean_addr, words)
            if address_type:
                address_types.append(address_type)

            for category, keywords in self.area_types.items():
                cn_matches = [w for w in words if any(k in w for k in keywords['cn'])]
                en_matches = [w for w in words if any(k in w.lower() for k in keywords['en'])]

                keyword_stats[category]['cn'].extend(cn_matches)
                keyword_stats[category]['en'].extend(en_matches)

        result = {}
        for category in self.area_types.keys():
            cn_counter = Counter(keyword_stats[category]['cn'])
            en_counter = Counter(keyword_stats[category]['en'])

            result[category] = {
                'chinese_keywords': dict(cn_counter.most_common()),
                'english_keywords': dict(en_counter.most_common()),
                'total_occurrences': len(keyword_stats[category]['cn']) + len(keyword_stats[category]['en'])
            }

        result['address_type_distribution'] = dict(Counter(address_types))

        return result

    def classify_address(self, address: str, words: List[str]) -> Optional[str]:
        """
        基于关键词对地址进行分类，返回地址类型。如果未能匹配到任何预定义类型，则返回 '未分类'。
        """
        scores = {category: 0 for category in self.area_types.keys()}

        for category, keywords in self.area_types.items():
            for word in words:
                if any(k in word for k in keywords['cn']):
                    scores[category] += 1  
                if any(k in word.lower() for k in keywords['en']):
                    scores[category] += 2

        if any(scores.values()): # 如果有任何区域类型得分大于0，则返回得分最高的类型
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return '未分类' # 简化后的版本，直接返回 '未分类'

def load_data(csv_file_path: str) -> pd.DataFrame:
    """
    加载 CSV 数据文件。
    """
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"成功读取CSV文件，共 {len(df)} 条记录，路径: {csv_file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"文件未找到: {csv_file_path}")
        raise  # 重新抛出异常，让主程序处理

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理数据，包括时间特征提取、地址类型分类和邻域类型校验。
    修改: 仅筛选 damf 非零值,保留所有原始列。
    """
    # 首先筛选出 damf 非零的行,但保留所有列
    if 'damf' in df.columns:
        df = df[df['damf'] != 0].copy()
        logging.info(f"已筛选 damf 非零行，剩余 {len(df)} 条记录。")
    else:
        logging.warning("未找到 'damf' 列，将继续处理所有数据。")

    # 初始化地址分析器
    analyzer = AddressAnalyzer()
    logging.info("地址分析器初始化完成。")

    # 时间特征提取
    df['tropomi_time'] = pd.to_datetime(df['tropomi_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce') # 增加错误处理
    if df['tropomi_time'].isnull().any():
        logging.warning("部分 'tropomi_time' 列转换失败，请检查时间格式。")
    else:
        logging.info("已将 'tropomi_time' 列转换为 datetime 格式。")

    # 标记星期天/工作日
    df['day_of_week'] = df['tropomi_time'].dt.dayofweek
    df['day_type'] = df['day_of_week'].apply(lambda x: '星期天' if x == 6 else '工作日')
    logging.info("已标记星期天/工作日信息 (day_type)。")

    # 标记季节信息
    def get_season(month):
        if 3 <= month <= 5:
            return '春季'
        elif 6 <= month <= 8:
            return '夏季'
        elif 9 <= month <= 11:
            return '秋季'
        elif 12 == month or 1 <= month <= 2: # 冬季包含12月，1月，2月
            return '冬季'
        else:
            return '秋季' # 9-11月为秋季

    df['season'] = df['tropomi_time'].dt.month.apply(get_season)
    logging.info("已标记季节信息 (season)。")

    # 标记节假日信息 (数据为美国地区)
    try:
        us_holidays = holidays.US()
        df['is_holiday'] = df['tropomi_time'].dt.date.apply(lambda date: date in us_holidays)
        logging.info("已标记美国节假日信息 (is_holiday)。")
    except Exception as e:
        logging.error(f"节假日信息处理出错: {e}，请检查 holidays 库是否正确安装以及地区设置。")
        df['is_holiday'] = False # 错误发生时，默认设置为 False，避免程序中断

    # 初步地址类型分类
    logging.info("开始进行初步地址类型分类...")
    address_classes_preliminary = [] # 存储初步分类结果
    for index, row in df.iterrows():
        address = analyzer.get_address(row['latitude'], row['longitude'])
        if address:
            clean_addr = analyzer.clean_address(address)
            words = analyzer.extract_keywords(clean_addr)
            address_class = analyzer.classify_address(clean_addr, words) # 调用原始的关键词分类函数
            address_classes_preliminary.append(address_class if address_class else '未分类')
        else:
            address_classes_preliminary.append('地址获取失败')
        if (index + 1) % 100 == 0:
            logging.info(f"已初步分类 {index+1}/{len(df)} 条记录")

    df['class_preliminary'] = address_classes_preliminary # 存储初步分类结果列
    logging.info("初步地址类型分类完成，已添加 'class_preliminary' 列。")

   # 直接使用初步分类结果作为最终分类结果
    df['class'] = df['class_preliminary']
    logging.info("已直接将初步分类结果作为最终分类结果。")

    df.drop(columns=['class_preliminary'], inplace=True) # 删除临时列 'class_preliminary'
    logging.info("已删除临时列 'class_preliminary'。")

    return df

def map_wavelength_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将光谱数据的列名从序号(0~496)映射为实际波长值(405~500nm)。
    同时处理radiance和irradiance光谱列。
    这是一个简单的列名映射，不改变数据本身。
    """
    # 计算波长范围和间隔
    start_wavelength = 405.0
    end_wavelength = 500.0
    num_wavelengths = 497
    wavelength_step = (end_wavelength - start_wavelength) / (num_wavelengths - 1)
    
    # 生成对应的波长值
    wavelengths = [start_wavelength + i * wavelength_step for i in range(num_wavelengths)]
    
    # 创建列名映射字典
    column_mapping = {}
    
    # 同时处理radiance和irradiance列
    for i, wavelength in enumerate(wavelengths):
        # 处理radiance列
        old_radiance = f'radiance_wavelength_{i}'
        new_radiance = f'radiance_{wavelength:.1f}nm'
        column_mapping[old_radiance] = new_radiance
        
        # 处理irradiance列
        old_irradiance = f'irradiance_wavelength_{i}'
        new_irradiance = f'irradiance_{wavelength:.1f}nm'
        column_mapping[old_irradiance] = new_irradiance
    
    # 获取所有需要处理的列（确保它们都存在于DataFrame中）
    columns_to_process = [col for col in column_mapping.keys() if col in df.columns]
    
    # 创建新的DataFrame仅包含我们想要的列并应用映射
    new_df = df[columns_to_process].copy()
    new_df = new_df.rename(columns=column_mapping)
    
    logging.info("radiance和irradiance光谱数据列名已映射到实际波长值。")
    return new_df

def analyze_location_types(df: pd.DataFrame, analyzer: AddressAnalyzer, sample_size: int = 2000) -> Dict:
    """
    分析地址类型，并返回关键词分析结果和地址类型分布。（此函数在分类功能移动到 preprocess_data 后，主要用于分析样本地址的关键词分布）
    """
    logging.info("\n开始收集地址样本进行关键词分析...") # 修改日志信息，表明用于关键词分析
    sample_size = min(sample_size, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    addresses = []
    for idx, row in enumerate(sample_df.iterrows()):
        address = analyzer.get_address(row[1]['latitude'], row[1]['longitude'])
        if address:
            addresses.append(address)
        if (idx + 1) % 100 == 0:
            logging.info(f"已处理 {idx+1}/{sample_size} 条记录用于关键词分析") # 修改日志信息

    logging.info("\n开始分析关键词...")
    analysis_results = analyzer.analyze_keywords(addresses)
    logging.info("\n关键词分析完成。")
    return analysis_results

def save_analysis_results(analysis_results: Dict, addresses: List[str], output_path: str, distribution_output_path: str):
    """
    保存分析结果到 Excel 文件。
    """
    analysis_df = pd.DataFrame([
        {
            'category': cat,
            'keyword_type': 'chinese' if k_type == 'chinese_keywords' else 'english',
            'keyword': kw,
            'count': count
        }
        for cat, data in analysis_results.items()
        if cat != 'address_type_distribution'
        for k_type in ['chinese_keywords', 'english_keywords']
        for kw, count in data[k_type].items()
    ])

    distribution_df = pd.DataFrame([
        {'type': type_name, 'count': count, 'percentage': (count / len(addresses)) * 100}
        for type_name, count in analysis_results['address_type_distribution'].items()
    ])

    with pd.ExcelWriter(output_path) as writer_keywords, pd.ExcelWriter(distribution_output_path) as writer_distribution: # 使用不同的 ExcelWriter 对象
        analysis_df.to_excel(writer_keywords, sheet_name='Keywords', index=False) # 保存关键词分析结果
        logging.info(f"关键词分析结果已保存至：{output_path}") # 关键词分析结果保存路径
        distribution_df.to_excel(writer_distribution, sheet_name='Type Distribution', index=False) # 保存地址类型分布
        logging.info(f"地址类型分布已保存至：{distribution_output_path}") # 地址类型分布保存路径
        pd.DataFrame({'sample_addresses': addresses}).to_excel(writer_keywords, sheet_name='Sample Addresses', index=False) # 保存样本地址到关键词分析结果文件中，方便查看
        logging.info(f"样本地址已保存至：{output_path} - Sheet: Sample Addresses") # 样本地址保存路径

def main():
    """
    主程序入口，执行数据加载、预处理（包含地址类型分类）、光谱重采样和结果保存。
    """
    print("程序开始执行...") # 打印程序开始信息，使用 print 确保用户看到

    csv_file_path = r"M:\Data_Set.csv"
    output_excel_path = r"D:\DBN\反射率\processed_result_with_neighbor_refine.xlsx" # 处理后的数据输出路径，修改文件名
    keywords_analysis_output_path = r"D:\address_keywords_analysis_results_neighbor_refine.xlsx" # 关键词分析结果输出路径，修改文件名以区分
    type_distribution_output_path = r"D:\address_type_distribution_results_neighbor_refine.xlsx" # 地址类型分布结果输出路径，新增路径

    try:
        df = load_data(csv_file_path)
        processed_df = preprocess_data(df.copy()) # 预处理数据，包含地址类型分类 (包含邻域校验)
        resampled_spectral_df = map_wavelength_columns(df.copy()) # 光谱映射，同时处理radiance和irradiance

        # 创建一个列表，包含所有需要排除的原始光谱列
        original_spectral_columns = []
        for i in range(497):
            original_spectral_columns.append(f'radiance_wavelength_{i}')
            original_spectral_columns.append(f'irradiance_wavelength_{i}')
            
        # 合并处理后的数据和重采样数据，排除原始光谱列
        output_df = pd.concat([
            processed_df.drop(columns=original_spectral_columns, errors='ignore'), 
            resampled_spectral_df
        ], axis=1)
        
        save_processed_data_excel_path = output_excel_path # 保存处理后数据的路径
        try:
            output_df.to_excel(save_processed_data_excel_path, index=False) # 保存处理后的数据，包含 'class' 列
            logging.info(f"处理结果已保存到 Excel 文件: {save_processed_data_excel_path}，包含 'class' 列和映射后的光谱数据") # 更新日志信息
        except Exception as e:
            logging.error(f"保存处理后数据 Excel 文件时出错: {e}")

        analyzer = AddressAnalyzer()  #  地址分析器实例
        analysis_results = analyze_location_types(df, analyzer) # 地点类型关键词分析 (针对样本地址)
        save_analysis_results(analysis_results, analyzer.cache.values(), keywords_analysis_output_path, type_distribution_output_path) # 保存地址关键词分析和类型分布结果，传递类型分布结果输出路径


        print("数据处理、地址分类和关键词分析完成 (包含邻域校验)。结果已保存到 Excel 文件。") # 使用 print 确保用户看到完成信息, 修改提示信息

        # 打印一些示例地址和分类结果
        print("\n地址样本示例 (前10条) 及分类结果：")
        sample_addresses_classified = [] # 存储地址和分类结果的列表
        sample_df_output = processed_df[['latitude', 'longitude', 'class']].sample(n=10, random_state=42) # 从 processed_df 中取样本
        for index, row in sample_df_output.iterrows():
            address = analyzer.get_address(row['latitude'], row['longitude']) # 重新获取地址 (虽然缓存中应该有，但为了代码完整性)
            if address:
                sample_addresses_classified.append(f"地址: {address}, 分类: {row['class']}") # 添加地址和分类结果
                print(f"  地址: {address}, 分类: {row['class']}") # 打印地址和分类结果
            else:
                sample_addresses_classified.append(f"经纬度: ({row['latitude']}, {row['longitude']}), 地址获取失败, 分类: {row['class']}") # 地址获取失败的情况
                print(f"  经纬度: ({row['latitude']}, {row['longitude']}), 地址获取失败, 分类: {row['class']}") # 打印地址获取失败信息

        # 打印关键词分析结果到控制台 (与之前相同)
        print("\n关键词分析结果 (部分类别)：") # 修改打印信息，表明是部分类别
        categories_to_print = ['教育区', '商业区', '工业区', '农业区', '交通枢纽', '公共设施', '居住区', '生态保护区','医疗健康区','军事区','文旅区','郊区','市区','科教研发区','重工业区','港口工业区','机场区域','高密度交通区','能源生产区' ]# 打印所有类别
        for category in categories_to_print: # 遍历部分类别
            if category in analysis_results and category != 'address_type_distribution': # 确保类别存在且不是地址类型分布
                data = analysis_results[category]
                print(f"\n{category}:")
                print("  中文关键词：")
                for word, count in data['chinese_keywords'].items():
                    print(f"    {word}: {count}次")
                print("  英文关键词：")
                for word, count in data['english_keywords'].items():
                    print(f"    {word}: {count}次")
                print(f"  总计出现次数: {data['total_occurrences']}")

        print("\n地址类型分布 (简要)：") # 修改打印信息，表明是简要信息
        if 'address_type_distribution' in analysis_results:
            for type_name, count in analysis_results['address_type_distribution'].items():
                percentage = (count / len(analyzer.cache)) * 100 # 使用缓存地址数量计算百分比
                if percentage > 1: # 只打印百分比大于 1% 的类型，避免信息过多
                    print(f"  {type_name}: {count} 次 ({percentage:.1f}%)")

    except FileNotFoundError:
        print("主程序：CSV 文件未找到，程序终止。") # 使用 print 提示文件未找到
        exit()
    except Exception as e:
        logging.error(f"主程序运行出错: {e}", exc_info=True) # 记录更详细的错误信息
        print(f"程序运行过程中发生错误，请查看日志文件。") # 使用 print 提示用户查看日志
    finally:
        print("程序执行完毕。") # 使用 print 提示程序结束

if __name__ == "__main__":
    main()