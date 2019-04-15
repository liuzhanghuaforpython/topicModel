
# coding: utf-8

# In[2]:


#step1：读取文件夹，遍历文件夹下的所有文件(去掉空文件)，写入result.txt文件下
import os
import jieba
path="C:\\Users\\tj\\jupyter\\news" #文件夹目录
files= os.listdir(path)  #返回路径下所有文件名字的列表
k=open('C:\\Users\\tj\\Desktop\\combine.txt','w+',encoding='utf-8')
for file in files: #遍历文件夹
    #print(file)
    f = open(path+"\\"+file,'r',encoding='gbk')#打开文件
    if os.path.getsize(path+"\\"+file)==0: #略过空文件
        pass
    else:
        for line in f:
            line=line.strip()
            k.write(line)
k.close()

#同义词表
syn=dict()
with open('C:\\Users\\tj\\Desktop\\synonyms.txt','r',encoding='utf-8')as f:
    for line in f:
        for word in line.strip('\n').split('\t')[1:]:
            syn[word]=line.strip('\n').split('\t')[0]
z=open('C:\\Users\\tj\\Desktop\\combine_syn.txt','w+',encoding='utf-8')
with open('C:\\Users\\tj\\Desktop\\combine.txt','r',encoding='utf-8')as f:
    for line in f:
        jieba.load_userdict('C:\\Users\\tj\\Desktop\\userdict1.txt')
        seg_list=jieba.cut(line.strip())
        for word in seg_list:
            if word in syn.keys():
                word=syn[word]
                z.write(word)
            else:
                z.write(word)
z.close()   

import jieba.posseg
jieba.enable_parallel(4)
pos=['n','eng','ns','x','nr','j','a','nt','l','b','nz','nrt','i']
d=open('C:\\Users\\tj\\Desktop\\combine_pos.txt','w+',encoding='utf-8')
f=open('C:\\Users\\tj\\Desktop\\combine_syn.txt','r',encoding='utf-8')
words=f.read()
jieba.load_userdict('C:\\Users\\tj\\Desktop\\userdict1.txt')
words_cut=jieba.posseg.cut(words) 
for i in words_cut:
    if i.flag in pos:
        d.write(i.word)
d.close()

#step2:对result.txt加载停用词表和自定义词典,提取topN个文本关键词，并将结果存入txt文件
import jieba
import jieba.analyse
import re
wf = open('C:\\Users\\tj\\Desktop\\hotwords_autoshow.txt','w+',encoding='utf-8')
for line in open('C:\\Users\\tj\\Desktop\\combine_pos.txt','r',encoding='utf-8'):
    item = line.strip('\n').split('\t')   #item[0]:str
    jieba.load_userdict('C:\\Users\\tj\\Desktop\\userdict1.txt')
    jieba.analyse.set_stop_words('C:\\Users\\tj\\Desktop\\stopwords_for_all_hotwords1.txt')
#    item[0]= re.sub('\d+', '', item[0]) 正则去掉数字
    tags = jieba.analyse.extract_tags(''.join(item[0]),500) #基于TF-IDF算法的关键词抽取,返回几个TF/IDF权重最大的关键词
#jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
#allowPOS 仅包括指定词性的词，默认值为空，即不筛选
    tagsw = ",".join(tags)
    wf.write(tagsw)
wf.close()

#生成关键词的字典
wf1=open('C:\\Users\\tj\\Desktop\\hotwords_autoshow.txt',encoding='utf-8').read()
wf1=wf1.split(',')#list
listVal = [];
for i in range(len(wf1)):
    listVal.append(0)
len(listVal)
word_dict={k:v for k,v in zip(wf1,listVal)}



k21=open('C:\\Users\\tj\\Desktop\\combine_syn.txt','r',encoding='utf-8')
words = k21.read().strip('\n').split('\t') #words是只有一个元素的列表
words_cut=jieba.cut(words[0])
result=','.join(words_cut) #result是str
result=result.split(',')   #str转list


#统计词频
for i in result:
    if i in word_dict:
        word_dict[i] +=1
    
import operator
sorted_word_dict=sorted(word_dict.items(),key=operator.itemgetter(1),reverse = True)    #按value排序，reverse = True--降序
#转化为list嵌套tuple
with open('C:\\Users\\tj\\Desktop\\autoshow_fre.txt','w')as k12:
    for item in sorted_word_dict:
        k12.write(item[0]+' '+str(item[1])+'\n')
k12.close()

import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

jieba.load_userdict('C:\\Users\\tj\\Desktop\\userdict1.txt')
#backgroud_Image = np.array(Image.open("C:\\Users\\tj\\Desktop\\car.png"))
#mask=backgroud_Image,
my_wordcloud=WordCloud(background_color='white',
                       font_path="C:\\Users\\tj\\Desktop\\ccnu\\simsun.txt",max_font_size = 75,random_state=30).generate_from_frequencies(word_dict) 
#用pyplot展示词云图
get_ipython().run_line_magic('pylab', 'inline')
plt.imshow(my_wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()
my_wordcloud.to_file("C:\\Users\\tj\\Desktop\\1.5wnews.png")

