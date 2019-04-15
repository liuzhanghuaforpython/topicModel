
# coding: utf-8

# In[ ]:


# 分别取涉及宝马、荣威、名爵、MG、大通、上汽大众的新闻
import os
import numpy as np
import pandas as pd

corpus_path = "news_data/by_date/20180828/"
file_list = os.listdir(corpus_path)
df_brandgeti = pd.read_csv('brandgeti.csv', encoding='gb18030')
df_brandgeti = df_brandgeti.dropna()
names = np.array(df_brandgeti)
names = names.tolist()
for name in names:
    os.mkdir("news_data/by_brand/" + name[0] + "/20180828")
    i = 1
    for file_name in file_list:
        full_name = corpus_path + file_name
        file_read = open(full_name, "r", encoding="utf8")
        content = file_read.read()
        if name[0] in content:
            i_str = str(i)
            filename = "news_data/by_brand/{}/20180828/".format(
                name[0]) + i_str + ".txt"
            outputs = open(filename, 'w', encoding='utf-8')
            outputs.write(content)
            outputs.close()
            i = i + 1
    print(name[0])

# 取涉及成都车展的新闻

name = "成都车展"
try:
    os.makedirs("news_data/by_chengdu/exhibition_before/20180828")
except:
    pass
i = 1
for file_path in file_list:
    full_name = corpus_path + file_path
    file_read = open(full_name, "r", encoding="utf8")
    content = file_read.read()

    if name in content:
        i_str = str(i)
        filename = "news_data/by_chengdu/exhibition_before/20180828/" + i_str + ".txt"
        outputs = open(filename, 'w', encoding='utf-8')
        outputs.write(content)
        outputs.close()
        i = i + 1

# 名爵MG新闻合并

basePATH = "news_data/by_brand/"
fileLis1 = os.listdir(basePATH + "名爵/20180828/")
fileLis2 = os.listdir(basePATH + "MG/20180828/")
fileLis = fileLis1 + fileLis2
lis = []

for file in fileLis1:
    string = open(basePATH + "名爵/20180828/" + file,
                  "r", encoding='utf-8').read()
    lis.append(string)
for file in fileLis2:
    string = open(basePATH + "MG/20180828/" + file,
                  "r", encoding='utf-8').read()
    lis.append(string)
try:
    os.makedirs(basePATH + "名爵MG/20180828")
except Exception as e:
    print(e)

for i, content in enumerate(list(set(lis))):
    with open(basePATH + "名爵MG/20180828/" + str(i) + ".txt", "w", encoding='utf-8') as f:
        f.write(content)
