import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
from sklearn.svm import SVR
from scipy.optimize import fsolve


df_ferrara = pd.read_csv('ferrara_270615.csv')
df_milano = pd.read_csv('milano_270615.csv')
df_mantova = pd.read_csv('mantova_270615.csv')
df_ravenna = pd.read_csv('ravenna_270615.csv')
df_torino = pd.read_csv('torino_270615.csv')
df_asti = pd.read_csv('asti_270615.csv')
df_bologna = pd.read_csv('bologna_270615.csv')
df_piacenza = pd.read_csv('piacenza_270615.csv')
df_cesena = pd.read_csv('cesena_270615.csv')
df_faenza = pd.read_csv('faenza_270615.csv')

#取出我们要分析的温度和日期数据
y1 = df_milano['temp']
x1 = df_milano['day']

#把日期数据转换 datetime的格式
day_milano = [parser.parse(x) for x in x1]

#调用subplot 函数 ，fig 是图像对象，ax是坐标轴对象
fig,ax = plt.subplots()

#调整x轴坐标制度，使其旋转70度，方便查看
plt.xticks(rotation = 70)

#设定时间格式

hours = mdates.DateFormatter('%H:%M')

#设定X轴显示的格式

ax.xaxis.set_major_formatter(hours)

#画出图像，day_milano是x轴数据，y1是Y轴数据，‘r’代表的是‘red’红色
ax.plot(day_milano,y1,'r')
# plt.show()
y1 = df_ravenna['temp']
x1 = df_ravenna['day']
y2 = df_faenza['temp']
x2 = df_faenza['day']
y3 = df_cesena['temp']
x3 = df_cesena['day']
y4 = df_milano['temp']
x4 = df_milano['day']
y5 = df_asti['temp']
x5 = df_asti['day']
y6 = df_torino['temp']
x6 = df_torino['day']

#把日期从string类型转化为标准的datatime类型
day_ravenna = [parser.parse(x) for x in x1]
day_faenza = [parser.parse(x) for x in x2]
day_cesena = [parser.parse(x) for x in x3]
day_milano = [parser.parse(x) for x in x4]
day_asti = [parser.parse(x) for x in x5]
day_torino = [parser.parse(x) for x in x6]

#调用subplot()函数，重新定义fig，ax变量
fig,ax = plt.subplots()
plt.xticks(rotation = 70)

hours = mdates.DateFormatter('%H:%H')
ax.xaxis.set_major_formatter(hours)

#这里需要画出三根线，所以需要三组参数，‘g’代表‘green’
ax.plot(day_ravenna,y1,'r',day_faenza,y2,'r',day_cesena,y3,'r')
ax.plot(day_milano,y4,'g',day_asti,y5,'g',day_torino,y6,'g')
plt.show()
#dist是一个装城市距离距离海边距离的列表
dist = [df_ravenna['dist'][0],
        df_cesena['dist'][0],
        df_faenza['dist'][0],
        df_ferrara['dist'][0],
        df_bologna['dist'][0],
        df_mantova['dist'][0],
        df_piacenza['dist'][0],
        df_milano['dist'][0],
        df_asti['dist'][0],
        df_torino['dist'][0]
        ]

#temp_max是一个存放每一个城市最高温度的列表
temp_max = [df_ravenna['temp'].max(),
    df_cesena['temp'].max(),
    df_faenza['temp'].max(),
    df_ferrara['temp'].max(),
    df_bologna['temp'].max(),
    df_mantova['temp'].max(),
    df_piacenza['temp'].max(),
    df_milano['temp'].max(),
    df_asti['temp'].max(),
    df_torino['temp'].max()
]
# temp_min 是一个存放每个城市最低温度的列表
temp_min = [df_ravenna['temp'].min(),
    df_cesena['temp'].min(),
    df_faenza['temp'].min(),
    df_ferrara['temp'].min(),
    df_bologna['temp'].min(),
    df_mantova['temp'].min(),
    df_piacenza['temp'].min(),
    df_milano['temp'].min(),
    df_asti['temp'].min(),
    df_torino['temp'].min()
]

fig,ax = plt.subplots()
ax.plot(dist,temp_max,'ro')
plt.show()


#用线性回归算法得到两条直线，分别表示两种不同的气温趋势，这样做很有趣。我们可以使用 scikit-learn 库的 SVR 方法。
# dist1是靠近海的城市集合，dist2是远离海洋的城市集合
dist1 = dist[0:5]
dist2 = dist[5:10]

# 改变列表的结构，dist1现在是5个列表的集合
# 之后我们会看到 nbumpy 中 reshape() 函数也有同样的作用
dist1 = [[x] for x in dist1]
dist2 = [[x] for x in dist2]

# temp_max1 是 dist1 中城市的对应最高温度
temp_max1 = temp_max[0:5]
# temp_max2 是 dist2 中城市的对应最高温度
temp_max2 = temp_max[5:10]

# 我们调用SVR函数，在参数中规定了使用线性的拟合函数
# 并且把 C 设为1000来尽量拟合数据（因为不需要精确预测不用担心过拟合）
svr_lin1 = SVR(kernel='linear', C=1e3)
svr_lin2 = SVR(kernel='linear', C=1e3)

# 加入数据，进行拟合（这一步可能会跑很久，大概10多分钟，休息一下:) ）
svr_lin1.fit(dist1, temp_max1)
svr_lin2.fit(dist2, temp_max2)

# 关于 reshape 函数请看代码后面的详细讨论
xp1 = np.arange(10,100,10).reshape((9,1))
xp2 = np.arange(50,400,50).reshape((7,1))
yp1 = svr_lin1.predict(xp1)
yp2 = svr_lin2.predict(xp2)

#限制了X轴的取值范围
ax.plot(xp1,yp1,c = 'b',label='Strong sea effect')
ax.plot(xp2,yp2,c = 'g',label='Light sea effect')

fig

print(svr_lin1.coef_) #斜率
print(svr_lin1.intercept_)  #截距
print(svr_lin2.coef_)
print(svr_lin2.intercept_)

# 定义了第一条拟合直线
def line1(x):
    a1 = svr_lin1.coef_[0][0]
    b1 = svr_lin1.intercept_[0]
    return a1*x + b1

# 定义了第二条拟合直线
def line2(x):
    a2 = svr_lin2.coef_[0][0]
    b2 = svr_lin2.intercept_[0]
    return a2*x + b2

# 定义了找到两条直线的交点的 x 坐标的函数
def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

result = findIntersection(line1,line2,0.0)
print("[x,y] = [ %d , %d ]" % (result,line1(result)))

# x = [0,10,20, ..., 300]
x = np.linspace(0,300,31)
plt.plot(x,line1(x),x,line2(x),result,line1(result),'ro')
plt.show()
#axis 函数规定了x轴和y轴的取值范围
plt.axis((0,400,15,25))
# plt.plot
plt.plot(dist,temp_min,'bo')
plt.show()

#5.2适度数据分析

#读取湿度数据
y1 = df_ravenna['humidity']
x1 = df_ravenna['day']
y2 = df_faenza['humidity']
x2 = df_faenza['day']
y3 = df_cesena['humidity']
x3 = df_cesena['day']
y4 = df_milano['humidity']
x4 = df_milano['day']
y5 = df_asti['humidity']
x5 = df_asti['day']
y6 = df_torino['humidity']
x6 = df_torino['day']

#重新定义fig和ax变量
fig,ax = plt.subplots()
plt.xticks(rotation = 70)

#把时间从string类型转化为标准的datatime类型
day_ravenna = [parser.parse(x) for x in x1]
day_faenza = [parser.parse(x) for x in x2]
day_cesena = [parser.parse(x) for x in x3]
day_milano = [parser.parse(x) for x in x4]
day_asti = [parser.parse(x) for x in x5]
day_torino = [parser.parse(x) for x in x6]

#规定时间的表示方式
hours = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(hours)

#表示在图上
ax.plot(day_ravenna,y1,'r',day_faenza,y2,'r',day_cesena,y3,'r')
ax.plot(day_milano,y4,'g',day_asti,y5,'g',day_torino,y6,'g')
plt.show()
# 获取最大湿度数据
hum_max = [df_ravenna['humidity'].max(),
df_cesena['humidity'].max(),
df_faenza['humidity'].max(),
df_ferrara['humidity'].max(),
df_bologna['humidity'].max(),
df_mantova['humidity'].max(),
df_piacenza['humidity'].max(),
df_milano['humidity'].max(),
df_asti['humidity'].max(),
df_torino['humidity'].max()
]

plt.plot(dist,hum_max,'bo')

plt.show()

#获取最小湿度
hum_min = [
df_ravenna['humidity'].min(),
df_cesena['humidity'].min(),
df_faenza['humidity'].min(),
df_ferrara['humidity'].min(),
df_bologna['humidity'].min(),
df_mantova['humidity'].min(),
df_piacenza['humidity'].min(),
df_milano['humidity'].min(),
df_asti['humidity'].min(),
df_torino['humidity'].min()
]
plt.plot(dist,hum_min,'bo')
plt.show()

#5.3 风向频率玫瑰图
# #在我们采集的每个城市的气象数据中，下面两个与风有关：
# 风力（风向）
# 风速
# 分析存放每个城市气象数据的 DataFrame 就会发现，风速不仅跟一天的时间段相关联，还与一个介于 0~360 度的方向有关。
#为了更好地分析这类数据，有必要将其做成可视化形式，但是对于风力数据，将其制作成使用笛卡儿坐标系的线性图不再是最佳选择。要是把一个 DataFrame 中的数据点做成散点图
#为了更好地分析这类数据，有必要将其做成可视化形式，但是对于风力数据，将其制作成使用笛卡儿坐标系的线性图不再是最佳选择。要是把一个 DataFrame 中的数据点做成散点图
plt.plot(df_ravenna['wind_deg'],df_ravenna['wind_speed'],'ro')
plt.show()

#要表示呈 360 度分布的数据点，最好使用另一种可视化方法：极区图。
# 首先，创建一个直方图，也就是将 360 度分为八个面元，每个面元为 45 度，把所有的数据点分到这八个面元中。

hist,bins = np.histogram(df_ravenna['wind_deg'],8,[0,360])
print(hist)
print(bins)
# histogram() 函数返回结果中的数组 hist 为落在每个面元的数据点数量。
# [0 5 11 1 0 1 0 0]
# 返回结果中的数组 bins 定义了 360 度范围内各面元的边界。
# [0. 45. 90. 135. 180. 225. 270. 315. 360.]
# 要想正确定义极区图，离不开这两个数组。我们将创建一个函数来绘制极区图。我们把这个函数定义为 showRoseWind()，它有三个参数：values 数组，指的是想为其作图的数据，也就是这里的 hist 数组；第二个参数 city_name 为字符串类型，指定图表标题所用的城市名称；最后一个参数 max_value 为整型，指定最大的蓝色值。
# 定义这样一个函数很有用，它既能避免多次重复编写相同的代码，还能增强代码的模块化程度，便于你把精力放到与函数内部操作相关的概念上。

def showRoseWind(values,city_name,max_value):
    N = 8

    # theta = [pi*1/4, pi*2/4, pi*3/4, ..., pi*2]
    theta = np.arange(0.,2 * np.pi, 2 * np.pi / N)
    radii = np.array(values)
    # 绘制极区图的坐标系
    plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

    # 列表中包含的是每一个扇区的 rgb 值，x越大，对应的color越接近蓝色
    colors = [(1-x/max_value, 1-x/max_value, 0.75) for x in radii]

    # 画出每个扇区
    plt.bar(theta, radii, width=(2*np.pi/N), bottom=0.0, color=colors)

    # 设置极区图的标题
    plt.title(city_name, x=0.2, fontsize=20)
    a= showRoseWind(hist, 'Ravenna', max(hist))
    plt.show(a)


    # plt.show()

# 由图 9-19 可见，整个 360 度的范围被分成八个区域（面元），每个区域弧长为 45 度，此外每个区域还有一列呈放射状排列的刻度值。在每个区域中，用半径长度可以改变的扇形表示一个数值，半径越长，扇形所表示的数值就越大。为了增强图表的可读性，我们使用与扇形半径相对应的颜色表。半径越长，扇形跨度越大，颜色越接近于深蓝色。
# 从刚得到的极区图可以得知风向在极坐标系中的分布方式。该图表示这一天大部分时间风都
# 吹向西南和正西方向。
# 定义好 showRoseWind() 函数之后，查看其他城市的风向情况也非常简单。
hist, bin = np.histogram(df_ferrara['wind_deg'],8,[0,360])
print(hist)
showRoseWind(hist,'Ferrara', max(hist))

#计算风速均值的分布情况
# 即使是跟风速相关的其他数据，也可以用极区图来表示。
# 定义 RoseWind_Speed 函数，计算将 360 度范围划分成的八个面元中每个面元的平均风速。

def RoseWind_Speed(df_city):
    # degs = [45, 90, ..., 360]
    degs = np.arange(45,361,45)
    tmp = []
    for deg in degs:
        # 获取 wind_deg 在指定范围的风速平均值数据
        tmp.append(df_city[(df_city['wind_deg']>(deg-46)) & (df_city['wind_deg']<deg)]
        ['wind_speed'].mean())
    return np.array(tmp)
# 这里 df_city[(df_city['wind_deg']>(deg-46)) & (df_city['wind_deg']<deg)] 获取的是风向大于'deg-46' 度和风向小于'deg' 的数据。
# RoseWind_Speed() 函数返回一个包含八个平均风速值的 NumPy 数组。该数组将作为先前定义的
# showRoseWind() 函数的第一个参数，这个函数是用来绘制极区图的。
showRoseWind(RoseWind_Speed(df_ravenna),'Ravenna',max(hist))