#--------------------------------教程------------------------------------
#https://mp.weixin.qq.com/s/FS70XOhmF2w4J3EghCzZfw
#--------------------------------导包------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                           #数据可视化包
import pyecharts
import missingno as msno                        #可视化查看丢失数据
plt.style.use('ggplot')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    #解决seaborn中文字体显示问题
plt.rc('figure', figsize = (20,20))             #把plt默认的图片size调大一点
%matplotlib inline

#--------------------------------数据倒入---------------------------------
'''
需求：用户想要了解实习僧上关于‘数据挖掘’、‘机器学习’、’算法’相关的岗位
分别倒入这三个关键词抓取的数据文件
'''
data_dm = pd.read_csv('datamining.csv')
data_ml = pd.read_csv('machinelearning.csv')
data_al = pd.read_csv('mlalgorithm.csv')
data = pd.concat([data_dm, data_ml, data_al], ignore_index = True)      #把三个data文件合并起来并且重新set索引

#--------------------------------数据基本处理---------------------------------
print(data.sample(3))       #随机查看3行数据

print(data.loc[3])          #抽取一条数据查看所有特征的数据类型

print(msno.matrix(data, labels=True))      #查看下数据缺失情况

print(data.shape)       #查看行列

print(data.info())      #查看字段基本情况

data.drop_duplicates(subset = 'job_links', inplace = True)  #根据招聘链接去重（(inplace=True)是直接对原dataFrame进行操作。）
print(data.shape)

我们可以看到原始数据的各字段都是object类型。接下来我们来看看整体的数据情况
#--------------------------------数据处理---------------------------------
#data。info（）函数可以查看字段的数据类型--》目前全部为object
#要将时间的字段改成datatime类型
#要对数字字段改成数值型数据
#com_id、com_links、com_location、com_website、com_welfare、detailed_intro、job_detail在本次分析中用不上，不处理
#注册号、招聘链接、公司地址、公司网站、公司福利、公司介绍、岗位介绍

#-------------------------------新建data_clean DataFrame------------------
#删除不需要的字段
data_clean = data.drop(['com_id','com_links','com_location','com_website','detailed_intro','job_detail'], axis =1 )


#-------------------------------数值型数据处理------------------------------
#包括：auth_capital, num_empliyee, wage, day_per_week, time_span
#逐一处理：
'''
auth_capital ----> 注册资本：1300万美元
处理思路:先把“注册资本：”清理掉，
再把数值型数据提取出来，
然后根据各币种的汇率，把注册资本转换为“万元人民币”单位
'''
#第一步：看下auth_capital的数据格式
data_clean.head()
data_clean.sample(4)

#第二步：切割字符串(：分割)，去除注册资本，保留金额和单位
auth_capital = data_clean['auth_capital'].str.split('：',expand = True)      #这里有2点要注意（1）expend = True代表，分拆后的各元素分别占一列 （2）'：'这里的冒号是中文格式的因为要和数据中的一致
auth_capital.sample(5)

#第三步：把金额提取出来，转换成float型
auth_capital['num'] = auth_capital[1].str.extract('([0-9.]+)', expand=False).astype('float')       #(1)将提取出的内容存入num字段 （2）'([0-9.]+)'为正则表达式 (3)dtype为查看数据，astype为转换数据

#第四步：处理币种单位
    #(1) 把金额和单位以‘万’split开，查看下有哪些币种
auth_capital[1].str.split('万', expand = True)[1].unique()   #split后金额存在‘0’字段，单位存在‘1’字段，对1字段做uniqe处理，就可以看出来有哪些币种

    #（2）定义判断是否为nan函数
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    #（3）定义函数，计算汇率
def get_ex_rate(string):
    if string is None:        #大大坑！！！！！这个错误的罪魁祸首-》》》》》 argument of type 'NoneType' is not iterable
        return np.nan
    if isfloat(string):      #大大坑！！！！！NaN是float格式！！！这个错误的罪魁祸首-》》》》》TypeError: argument of type 'float' is not iterable
        return np.nan
    if '人民币' in string:
        return 1.00
    elif '港' in string:
        return 0.80
    elif '美元' in string:
        return 6.29
    elif '欧元' in string:
        return 7.73
    else:
        return np.nan

#（4）把汇率存入ex_rate字段
auth_capital['ex_rate'] = auth_capital[1].apply(get_ex_rate)

    #（3）把金额统一为人民币
data_clean['auth_capital'] = auth_capital['num'] * auth_capital['ex_rate']

'''
day_per_week 字段处理
直接把字段中别的汉字去除，保留数字
'''
#第一步，看下数据
data_clean.day_per_week.unique()    #查看下该字段先有哪些数据（不重复的）

#第二步，直接赋值,用loc定位数据的行，列  ----》通过标签的label的形式选取数据---loc
data_clean.loc[data['day_per_week']=='2天/周', 'day_per_week'] = 2
data_clean.loc[data['day_per_week']=='3天/周', 'day_per_week'] = 3
data_clean.loc[data['day_per_week']=='4天/周', 'day_per_week'] = 4
data_clean.loc[data['day_per_week']=='5天/周', 'day_per_week'] = 5
data_clean.loc[data['day_per_week']=='6天/周', 'day_per_week'] = 6

'''
num_employee 字段处理
'少于15人'、'15-50人'、'50-150人'都记为'小型企业',
'150-500人'、'500-2000人'记为'中型企业'，
'2000人以上'、'5000人以上'记为大型企业
'''
#第一步，看下数据
data_clean.num_employee.unique()    #查看下该字段先有哪些数据（不重复的）

#第二步，直接赋值,用loc定位数据的行，列  ----》通过标签的label的形式选取数据---loc
data_clean.loc[data['num_employee'] == '少于15人', 'num_employee'] = '小型公司'
data_clean.loc[data['num_employee'] == '15-50人', 'num_employee'] = '小型公司'
data_clean.loc[data['num_employee'] == '50-150人', 'num_employee'] = '小型公司'
data_clean.loc[data['num_employee'] == '150-500人', 'num_employee'] = '中型公司'
data_clean.loc[data['num_employee'] == '500-2000人', 'num_employee' ] = '中型公司'
data_clean.loc[data['num_employee'] == '2000人以上', 'num_employee' ] = '大型公司'
data_clean.loc[data['num_employee'] == '5000人以上', 'num_employee' ] = '大型公司'


'''
time_span 实习时长，字段处理

'''
#第一步，看下数据
data_clean.time_span.unique()    #查看下该字段先有哪些数据（不重复的）


#第二步，处理逻辑和上个字段一样，但是由于枚举过多，这次用函数形式处理
    #(构造一个字典，通过pd.Series.map() 也就是映射的方式来做，方便快捷)
    #(1) 用mapping做一组键值对
mapping = {}
for i in range(1,19):        #枚举值在1-18个月
    mapping[str(i) + '个月'] = i
print(mapping)
    #(2) 用map函数把键值对中的值赋到df中
data_clean['time_span'] = data['time_span'].map(mapping)



'''
wage 每天工资，字段处理
1. 查看后，该字段没有缺失值
'''
#第一步，看下数据
msno.matrix(data_clean, labels=True)
data_clean['wage'].sample(5)

#第二步，把工资区间变为取一个最低和最高工资
data['wage'].str.extract('([0-9.]+)-([0-9.]+)/天', expand = True)    #数据类型为object

#第三步，把数据类型变成int
data['wage'].str.extract('([0-9.]+)-([0-9.]+)/天', expand = True).astype('int')

#第四步，求平均值
data_clean['average_wage'] = data['wage'].str.extract('([0-9.]+)-([0-9.]+)/天', expand = True).astype('int').mean(axis = 1)

#-------------------------------时间数据处理------------------------------
'''
时间数据字段包括：
“est_date”（公司成立日期）、“job_deadline”（截止时间）、
“released_time”（发布时间）、“update_time”（更新时间）等字段
'''

'''
est_date 期望时间格式：xxxx-xx-xx
'''
#第一步：查看数据
data['est_date'].sample(5)      #现在是object类型

#第二步：正则提取
data['est_date'].str.extract('成立日期：([0-9-]+)', expand=False)        #现在是object类型

#第三步：数据类型转换成datetime
data_clean['est_date'] = pd.to_datetime(data['est_date'].str.extract('成立日期：([0-9-]+)', expand=False))
data_clean['est_date'].sample(5)

'''
job_deadline
查看后，数据很感觉，只需要把oject转换成datetime即可
'''
#第一步：查看数据
data['job_deadline'].sample(5)      #现在是object类型

#第二步：数据转换
data_clean['job_deadline'] =pd.to_datetime(data['job_deadline'])


'''
released_time (发布时间)
查看后，数据很感觉，只需要把oject转换成datetime即可

该字段1小时内的都以分钟表示、
1小时-2天内的都以小时表示、
2天-1周内的都以天表示，
1周-1个月内的都以周表示，
处理逻辑：
可以考虑清洗成：2天以内是最新的（newest），
2天-1周是新的（new），
1周-1个月是可以投简历的（acceptable），
1个月以上的是旧的（old）

先把每条记录中的分钟、小时、天、周、月提取出来，再定义一个映射map一下就可以了
'''
#第一步：查看数据
data['released_time'].sample(5)

#第二步：数据处理
    #（1）正则提取处理数据
data['released_time'].str.extract('[0-9-]+(\w+)前', expand = True)
    # (2)用unique()查看枚举值
data['released_time'].str.extract('[0-9-]+(\w+)前', expand = False).unique()
    # (3) map映射处理，实例：‘小时’：‘newest’
data_clean['released_time'] = data['released_time'].str.extract('[0-9-]+(\w+)前', expand =False).map({
'分钟':'newest','小时':'newest','天':'new','周':'acceptable','月':'old'
})


'''
update_time
查看后，数据很感觉，只需要把oject转换成datetime即可
'''

#第一步：查看数据
data['update_time'].sample(5)

#第二步：数据转换
data_clean['update_time'] = pd.to_datetime(data['update_time'])


#------------------------------字符型数据处理------------------------------
'''
字段包括：
包括“city”（城市）、“com_class”（公司类型）、“com_intro（公司简介）”、“job_title”（职位名称）等字段
'''

'''
city 字段处理
数据相对干净，直接赋值处理
'''

#第一步：查看数据的枚举值
data['city'].unique()

#第二步：赋值处理
data_clean.loc[data_clean['city'] == '成都市', 'city'] = '成都'
data_clean.loc[data_clean['city'].isin(['珠海市','珠海 深圳']), 'city'] = '珠海'
data_clean.loc[data_clean['city'] == '上海漕河泾开发区','city'] = '上海'

#第三步：看下数据情况
msno.matrix(data_clean, labels=True)

#第四步：看看招聘“机器学习算法”实习生前10的城市 --------【【【待解决】】】
data_clean['city'].value_counts().nlargest(10).plot(kind = 'bar')

'''
com_class 公司及企业类型处理

'''

#第一步：查看数据的枚举值   --- 以列表形式展示
list(data['com_class'].unique())

#第二步：定义判断是否为nan函数
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
#第三步：定义拆分函数
def get_com_type(string):
    if string is None:        #大大坑！！！！！这个错误的罪魁祸首-》》》》》 argument of type 'NoneType' is not iterable
        return np.nan
    if isfloat(string):      #大大坑！！！！！NaN是float格式！！！这个错误的罪魁祸首-》》》》》TypeError: argument of type 'float' is not iterable
        return np.nan
    if ('非上市' in string) or ('未上市' in string):
        return '股份有限公司（未上市）'
    elif '股份' in string:
        return '股份有限公司（上市）'
    elif '责任' in string:
        return '有限责任公司'
    elif '外商投资' in string:
        return '外商投资公司'
    elif '有限合伙' in string:
        return '有限合伙企业'
    elif '全民所有' in string:
        return '国有企业'
    else:
        return np.nan

#第四步：拆分字段数据，并保存
com_class = data['com_class'].str.split('：', expand = True)  #实例数据： 公司类型：有限责任公司(台港澳法人独资)
com_class['com_class'] = com_class[1].apply(get_com_type)     #调用函数，统一数据
com_class.sample(5)

data_clean['com_class'] = com_class['com_class']    #把com_class的数据存入data——clean中

#对清理好的数据重新排序
data_clean = data_clean.reindex(columns=['com_fullname', 'com_name', 'job_academic', 'job_links',
                                        'tag','auth_capital', 'day_per_week', 'num_employee', 'time_span',
                                        'average_wage', 'est_date', 'job_deadline', 'released_time',
                                        'update_time', 'city', 'com_class', 'com_intro', 'job_title',
                                        'com_logo', 'industry'])

#--------------------------------【数据分析开始】---------------------------------
#查看清洗后的数据情况
data_clean.sample(3)
#看下数据类型
data_clean.info()
#调整一下数据类型
data_clean['update_time'] = data_clean['update_time'].astype('datetime64[ns]')
data_clean['auth_capital'] = data_clean['auth_capital'].astype('float64')

# 分析1: 城市和职位数量
#（1）查看排名前15的公司
city = data_clean['city'].value_counts()
city[:15]

# #(2) 画出柱状图，并展示最高值的值---------------------暂时没法解决
# from pyecharts import Bar
# bar = pyecharts.Bar('城市与职位数量')
# bar.add('', city[:15].index, city[:15].values, mark_point=['max'])
# bar
#
# # 用matplotlib直接画柱状图------------如何排序呢？？？？
# # 作图
city_jobs = plt.bar(city[:15].index, city[:15].values)
# #展示数据标签
for city in city_jobs:
     height = city.get_height()
     plt.text(city.get_x() + city.get_width() / 2, height, str(height), size=15, ha='center', va='bottom')



#（3） 排名前15的公司岗位占比
city_pct = (city/city.sum()).map(lambda x : '{:,.2%}'.format(x))    #map表示对一组x都进行操作
city_pct[:15]                                                       #'{:,.2%}'.format(x)输出百分比

#（4） 排名前5的公司岗位占总岗位的数量
(city/city.sum())[:5].sum()

# (5) 求出杭州市，岗位最多的5个公司
data_clean.loc[data_clean['city'] == '杭州', 'com_name'].value_counts()[:5]

# (6) 再看看这15个个城市中，招聘职位数量前五的公司到底是哪些
    # - 设置函数取出每个城市中前5的公司有哪些
def topN(dataframe, n = 5):
    counts = dataframe, value_counts()  #计算该字段相同数据出现的次数
    return counts[:5]

    # - 通过groupby 把大df以city纬度切成小df
data.clean.groupby('city').com_name.apply(topN).loc[list(city_pct[:15].index)]

list(city_pct[:5].index)  #------》['北京', '上海', '杭州', '深圳', '广州']
list(city_pct[:5])        #------》['53.89%', '18.76%', '6.29%', '5.95%', '5.03%']



data_clean['salary'] = data_clean['average_wage'] * data_clean['day_per_week'] * 4
data_clean['salary'].mean()

# 分析2: 薪资

#平均月薪
data_clean['salary'] = data_clean['average_wage'] * data_clean['day_per_week'] * 4
data_clean['salary'].mean()

#实习天数
data_clean['day_per_week'].unique()

#薪资与城市--查询平均实习月薪最高的10个城市
salary_by_city = data_clean.groupby('city')['salary'].mean()
salary_by_city.nlargest(10)

#岗位最多的10个城市平均实习月薪从大到小排列
top10_city = salary_by_city[city_pct[:10].index].sort_values(ascending = False)

# # 作图展示排名前10的城市的平均工资 --------用matplotlib直接画柱状图
city_y = top10_city.index
city_x = top10_city.values
city_salary = plt.bar(city_y, city_x)
plt.xticks(rotation =0)
for city in city_salary:
     height = city.get_height()
     height = np.around(height,2)
     plt.text(city.get_x() + city.get_width() / 2, height, str(height), size=10, rotation=0, ha='center', va='bottom')


# 查看城市实习薪资的分布情况
plt.figure(figsize=(15, 10))
top10_city_box = data_clean.loc[data_clean['city'].isin(top10_city.index),:]
sns.violinplot(x ='salary', y ='city', data = top10_city_box)


# 学历要求
job_academic =data_clean['job_academic'].value_counts()
job_academic

# 可以做饼图分析
plt.pie(job_academic.values, labels = job_academic.index, explode = (0, 0.1, 0, 0, 0), autopct='%1.1f%%',shadow=True, startangle=180, pctdistance=1.5,labeldistance=1.6)
plt.axis('equal')   #把饼图变成整圆


# 学历与薪资
# 求一下不同学历的平均工资
data_clean.groupby('job_academic')['salary'].mean().sort_values()

# 用箱式图
sns.boxplot(x = 'job_academic', y = 'salary', data = data_clean)
sns.violinplot(x ='job_academic', y ='salary', data = data_clean)

# 行业
# 需求：在现在的各行各业中，哪些行业对数据挖掘、机器学习算法的实习生需求更多，还有哪些行业现在也正在应用机器学习算法

# 处理数据
data_clean['industry'].sample(5)
list(data_clean['industry'].unique())

# 把数据用/ ， , 分割开
industry = data_clean.industry.str.split('/|,|，', expand = True)  #会切出来一个6column的df，因为最多的一个数据会包括5个行业

# 把前15需求量大的行业列出来
industry_top15 = industry.apply(pd.value_counts).sum(axis = 1).nlargest(15)
'''
拆解一下：
industry.apply(pd.value_counts) ----》 把之前拆出来的6列，把每列相同的字段计算出来
industry.apply(pd.value_counts).sum(axis = 1) ----》把每个字段6列统计出来的数加总
industry.apply(pd.value_counts).sum(axis = 1).nlargest(15) ----》再选出前15
'''

# 画出行业与岗位的数量
city_position = plt.bar(industry_top15.index, industry_top15.values)
plt.xticks(rotation =90)
for city in city_position:
     height = city.get_height()
     height = np.around(height.astype(int))  #把小数变成整数
     plt.text(city.get_x() + city.get_width() / 2, height, str(height), size=10, rotation=0, ha='center', va='bottom')


#公司
#需求：按照公司统计，发布招聘信息数量及平均工资
data_clean.groupby('com_name').salary.agg(['count', 'mean']).sort_values(by = 'count', ascending = False)[:15]

#公司规模与职位数量
data_clean['num_employee'].value_counts()

#公司规模与实习月薪
data_clean.groupby('num_employee').salary.mean().sort_values(ascending = False)

#公司实习期长度
data_clean['time_span'].value_counts()

#平均实习时长
data_clean['time_span'].mean()

#企业成立时间
#（1）通过公司去重，因为一个公司会出现多个职位
est_date = data_clean.drop_duplicates(subset = 'com_name')

#(2)把之前清理好的成立时间数据，再清理一下，以年做单位
import warnings
warnings.filterwarnings('ignore')   #----》防止报错
est_date['est_year'] = pd.DatetimeIndex(est_date['est_date']).year
'''
分解一下：
type(est_date['est_date']) ---->可以看出来est_date是series
type(pd.DatetimeIndex(est_date['est_date']) ----》将series转换成datatimeindex)
est_date['est_year'] = pd.DatetimeIndex(est_date['est_date']).year------>转换成年为单位
'''
#计算下每年成立的企业数量
num_com_by_year = est_date.groupby('est_year')['com_name'].count()

#画线性图看下---
plt.figure(figsize=(15, 10))
com_per_yr = plt.plot(num_com_by_year.index, num_com_by_year.values, color="darkblue",linewidth=1,linestyle='--',label='JJ income', marker='+')
for a,b in zip(num_com_by_year.index,num_com_by_year.values,):
     plt.text(a, b, b, ha='center', va='bottom', fontsize=15)


#那新成立的企业中，企业规模怎么样？
#把数据以企业规模和成立年份分组计数

scale_VS_year = est_date.groupby(['num_employee', 'est_year'])['com_name'].count()
scale_VS_year_s = scale_VS_year['小型公司'].reindex(num_com_by_year.index, fill_value=0)
scale_VS_year_m = scale_VS_year['中型公司'].reindex(num_com_by_year.index, fill_value=0)
scale_VS_year_l = scale_VS_year['大型公司'].reindex(num_com_by_year.index, fill_value=0)
'''
分解一下：
scale_VS_year['小型公司'] ----》把scale_VS_year groupby后的小公司df存入scale_VS_year_s
scale_VS_year['小型公司'].index ----》现在的index不是连续的年份，后期作图的话不对比
scale_VS_year['小型公司'].reindex(num_com_by_year.index, fill_value=0) -----》给数据重新定义检索，根据是num_com_by_year，如果检索对应的值没有，则填充为0

'''

# 做线性图看一下-------还没做
plt.figure(figsize=(15, 10))
plt.plot(scale_VS_year_s.index, scale_VS_year_s.values, color="darkblue",linewidth=1,linestyle='--',label='小型公司', marker='+')
plt.plot(scale_VS_year_m.index, scale_VS_year_m.values, color="lightblue",linewidth=1,linestyle='--',label='中型公司', marker='+')
plt.plot(scale_VS_year_l.index, scale_VS_year_l.values, color="lightgreen",linewidth=1,linestyle='--',label='大型公司', marker='+')
plt.legend()



# 保存清洗过的数据到csv
data_clean.to_csv('/Users/apple/Desktop/shixiseng/data_clean.csv', index = False)

#--------------------------------路径查询-------------------------------------
import  os
print os.getcwd() #获取当前工作目录路径
print os.path.abspath('.') #获取当前工作目录路径
print os.path.abspath('test.txt') #获取当前目录文件下的工作目录路径
print os.path.abspath('..') #获取当前工作的父目录 ！注意是父目录路径
print os.path.abspath(os.curdir) #获取当前工作目录路径




















#
