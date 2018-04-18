# Pandas底层调用很多是基于Numpy
# .csv文件：以逗号为分隔符的文件
# Numpy的核心结构是np.array

###
# 一段示例代码
###
import pandas
food_info = pandas.read_csv("food_info.csv")

print(type(food_info))                         # Pandas的核心结构是DataFrame, 包含几种常见的结构，                    
print(food_info.dtypes)                        # 比如object(字符型)，int64，float64，datetime, bool  

print(help(pandas.read_csv))                   

food_info.head()                               # 打印上面读出的数据显示出来，默认显示前5条，括号里面是参数
food_info.tail(4)                              # 打印数据的后4条
print(food_info.columns)                       # 打印出全部列名
print(food_info.shape)                         # !!!这条比较核心，可以打印整个样本的矩阵维度

# Pandas取数据相较于python或者np的直接指定idx来说要麻烦一些，需要依赖函数.loc
# 按行取数据
print(food_info.loc[0])                        # 打印出按行标记的第0个数据
print(food_info.loc[3:6])                      # 取出从3到6行
print(food_info.loc[2,5,10])                   # 取出2,5,10行

# 按列取数据
# Pandas默认第1行就是列名
ndb_col = food_info["NDB_NO"]                  # 取出列名为NDB_NO的整列
print(ndb_col)                                 # col_name = "NDB_NO" , ndb_col = food_info[col_name]

# 取出若干列
columns = ["Zinc_(mg)", "Copper_(mg)"]         # 取出列名为"Zinc_(mg)"和"Copper_(mg)"的两列
zinc_copper = food_info[columns]
print(zinc_copper)

# 取出列名中以"(g)"结尾的
col_names = food_info.columns.tolist()         # 把列名做成list结构，里面的元素就是全部的列名
print(col_names)
for c in col_names:
  if c.endswith("(g)"):
    gram_columns.append(c)
gram_df = food_info[gram_columns]
print(gram_df.head(3))

# Pandas对数据进行四则运算
print(food_info["Iron_(mg)"])                  # 把列名为"Iron_(mg)"的数都除以1000
div_1000 = food_info["Iron_(mg)"] / 1000
print(div_1000)

weigthed_protein = food_info["Protein_(g)"] * 2
weigthed_fat     = -0.75 * food_info["Lipid_Tot_(g)"]
initial_rating   = weigthed_protein + weigthed_fat

water_energy = food_info["Water_(g)"] * food_info["Energ_Kcal"]  # 这种维度一样的四则运算默认是对应位置的元素之间进行！类似于广播模式！
iron_grams   = food_info["Iron_(mg)"] / 1000
print(food_info.shape)                         # 打印结果:(8618, 36)
food_info["Iron_(g)"] = iron_grams             # 这里新建列名并且赋值
print(food_info.shape)                         # 打印结果:(8618, 37)

# Pandas求取最值，均值
max_calories = food_info["Energ_Kcal"].max()  
nomalized_calories = food_info["Energ_Kcal"] / max_calories                      # 归一化
nomalized_protein  = food_info["Protein_(g)"] / food_info["Protein_(g)"].max()   # 归一化

# 排序.sort_values()
food_info.sort_values("Sodium_(mg)", inplace=True)                      # 对Sodium_(mg)列进行排序，inplace表示排序好的列是替换 
                                                                        # 原列还是变成新列加入数据集。默认是从小到大的排列
food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)     # ascending=False表示降序排列。NaN表示缺失值，放在最后                                                                
###

###
# 另一段示例代码，数据集为titanic_train.csv
###
import pandas as pd
import numpy as np
titanic_survival = pd.read_csv("titanic_train.csv")    # 读入数据   
titanic_survival.head()                                # 打印数据集的前5行

age = titanic_survival["Age"]                          # "Age"为列名
age_is_null    = pd.isnull(age)                        # 拿出缺失值
age_null_count = len(age_is_null)                      # 计算缺失值个数
 
# 如果不对缺失值进行处理，那么在进行求取均值操作时候，结果就会是NaN,例子如下
mean_age = sum(titanic_survival["Age"]) / len(titanic_survival["Age"])      # 得到结果为NaN
# 增加了对缺失值的丢弃，代码如下
good_ages = titanic_survival["Age"][age_is_null == False]                   # 注意这里是两个方括号！！！！
correct_mean_age = sum(good_ages) / len(good_ages)
# 或者直接调用.mean()函数,也可以丢掉缺失值
correct_mean_age = titanic_survival["Age"].mean()     # 这个方法不太好，因为一般缺失值不会丢弃，可以用均值，中值，众值来填充

# 求取某等级的船票(Pclass)的均值(一个数据统计的工作)
passenger_classes = [1, 2, 3]
fares_by_class = {}
for this_class in passenger_classes:
  pclass_rows    = titanic_survival[titanic_survival["Pclass"] == this_class]   # 遍历，找到做某个舱的全部的人
  pclass_fares   = pclass_rows["Fare"]                                          # "Fare"表示船票价格的列
  fare_for_class = pclass_fares.mean()                                          # 求船票价格的均值
  fare_for_class[this_class] = fare_for_class                                   # 找到某等级的舱的船票各自的均值是多少
print(fare_for_class)  
# 上面的办法过于复杂，可以用.pivot_table()函数
# 下面计算的是某等级的船票(Pclass)获救的平均人数
passenger_survival = titanic_survival.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
# 下面计算的是某等级的船票(Pclass)的人的平均年龄
passenger_age = titanic_survival.pivot_table(index="Pclass", values="Age")      # 不指定aggfunc时，默认也是平均
# 下面计算的是某登船地点(Embarked)的人的船票价格(Fare)的总以及生还人数(Survived)的总和
port_stats = titanic_survival.pivot_table(index="Embarked", values=["Fare","Survived"], aggfunc=np.sum)  # 这里的是np.sum！

# .dropna()丢地缺失值
drop_na_columns = titanic_survival.dropna(axis=1)                               # axis=1/0按列/行丢弃
new_titanic_survival = titanic_survival.dropna(axis=0, subset=["Age","Sex"])    # 丢弃"Age"和"Sex"为缺失值的行！注意这里是行！

# 定位到具体的值.loc                         [行数， 列数]
row_index_83_age      = titanic_survival.loc[83, "Age"]              # 找到第83个样本的"Age"值
row_index_1000_pclass = titanic_survival.loc[76, "Pclass"]           # 找到第76个样本的"Pclass"值
  
# 重新生成索引.reset_index()
new_titanic_survival = titanic_survival.sort_values("Age"m ascengding=False)        # "Age"列降序排序
print(new_titanic_survival[0:10])                                                   # 打印这0-10个，结果正确但是索引却是原始的索引
titanic_reindexed = new_titanic_survival.reset_index(drop=True)                     # .reset_index可以重新生成索引
print('-----------------')
print(titanic_reindexed.loc[0:10])
  
# Pandas自定义函数.apply()
def hundredth_row(column):
  hundredth_item = column.loc[99]
  return hundredth_item
hundredth_row = titanic_survival.apply(hundredth_row)
print(hundredth_row)

def not_null_count(column):
  column_null = pd.isnull(column)
  null = column[column_null]
  return null
column_null_count = titanic_survival.apply(not_null_count)
print(column_null_count)

def which_class(row):
  pclass = row['Pclass']
  if pd.isnull(pclass):
    return "Unkown"
  elif pclass == 1:
    return "First Class"
  elif pclass == 2:
    return "Second Class"
  elif pclass == 3:
    return "Third Class"
classes = titanic_survival.apply(which_class, axis=1)            # axis = 1 --> 按列处理
print(classes)
  
def is_minor(row):
  if row["Age"] < 18:
    return True
  else:
    return False
minors = titanic_survival.apply(is_minor, axis=1)
print(minors)

def generate_age_label(row):
  age = row["Age"]
  if pd.isnull(age):
    return "unknown"
  elif age < 18:
    return "minor"
  else:
    reutnr "adult"
age_labels = titanic_survival.apply(generate_age_label, axis=1)
print(age_labels)

# 获救的人是否成年的比例
titanic_survival['age_labels'] = age_labels
age_group_survival = titanic_survival.pivot_table(index="age_labels", values="Survived")
print(age_group_survival)
###  
  
# Series结构
###
# Series结构是DataFrame(经过read_csv得到的)其中的一行或者一列
# DataFrame由Series构成；Series由np.array构成
###  
import pandas as pd
fandango    = pd.read_csv('fandango_score_comparison.csv')       # 这个数据集是对一些列电影的不同媒体的评分等等构成的
series_film = fandango['FILM'] 
print(type(series_film))
print(series_film[0:5])
series_rt = fandango['RottenTomatoes']
print(series_rt[0:5])

from pandas import Series
film_names = series_film.values
rt_scores = series_rt.values
series_custom = Series(rt_scores, index=film_names)
series_custom[['Minions (2015)', 'Leviathan (2014)']]
fiveten = series_custom[5:10]  

original_index = series_custom.index.tolist()
sorted_index = sorted(original_index)
sorted_by_index = series_custom.reindex(sorted_index)
  
sc2 = series_custom.sort_index()
sc3 = series_custom.sort_values()
print(sc3[0:10])

import numpy as np
print(np.add(series_custom, series_custom))
np.sin(series_custom)
np.max(series_custom)
  
series_custom > 50
series_greater_than_50 = series_custom[series_custom > 50]
criteria_one  = series_custom > 50
criteria_two  = series_custom < 75
both_criteria = series_custom[criteria_one & criteria_two] 
print(both_criteria)
  
rt_critics = Series(fandango['RottenTomatoes'].values, index=fandango['FILM'])     
rt_users   = Series(fandango['RottenTomatoes_User'].values, index=fandango['FILM'])
rt_mean    = (rt_critics + rt_users) / 2
print(rt_mean) 
###  

###
# 做一个Series结构
###
import pandas as pd

fandango = pd.read_csv('fandango_score_comparison.csv')
print(type(fandango))
fandango_films = fandango.set_index('FILM', drop=False)

fandango_films["Avengers: Age of Ultron (2015)" : "Hot Tub Time Mahine 2 (2015)"]
fandango_films.loc["Avengers: Age of Ultron (2015)" : "Hot Tub Time Mahine 2 (2015)"]
fandango_films.loc['Kumiko, The Treasure Hunter (2015)']
movies = ['Kumiko, The Treasure Hunter (2015)' , 'Do You Believe? (2015)' , 'Ant-Man (2015)']
###
