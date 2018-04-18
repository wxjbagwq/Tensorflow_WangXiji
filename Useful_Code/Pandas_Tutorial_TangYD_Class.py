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











###

