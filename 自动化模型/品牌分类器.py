
# coding: utf-8

# In[ ]:


#最终的品牌分类器
import pandas as pd
import numpy as np
import jieba 
# from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
# from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC

df1 = pd.read_csv('../input/fiveweb/ifeng.csv', encoding='gb18030')
df2 = pd.read_csv('Test.csv', encoding='utf8')

def chinese_word_cut(mytext):
    jieba.load_userdict('dictionary.txt')
    return " ".join(jieba.cut(mytext))

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

df1 = df1.dropna()
df2 = df2.dropna()
cn1 = df1.car_name
cn1 = np.array(cn1)
cn1 = cn1.tolist()
bn1 = df1.brand_name
bn1 = np.array(bn1)
bn1 = bn1.tolist()
cn2 = df2.car_name
cn2 = np.array(cn2)
cn2 = cn2.tolist()
bn2 = df2.brand_name
bn2 = np.array(bn2)
bn2 = bn2.tolist()

li1 = []
for i in range(0,len(bn1)):
    if bn1[i] not in cn1[i]:
        bc1 = bn1[i] + cn1[i]
    else:
        bc1 = cn1[i]
    li1.append(bc1)
li2 = []
for i in range(0,len(bn2)):
    if bn2[i] not in cn2[i]:
        bc2 = bn2[i] + cn2[i]
    else:
        bc2 = cn2[i]
    li2.append(bc2)
    
brand_and_car1 = pd.DataFrame(li1,columns=['brand_and_car1'])
brand_and_car2 = pd.DataFrame(li2,columns=['brand_and_car2'])
x_test = brand_and_car1
x_train = brand_and_car2
y_train = df2.standard_brand

x_train['cutted_brand_and_car'] = x_train.brand_and_car2.apply(chinese_word_cut)
x_test['cutted_brand_and_car'] = x_test.brand_and_car1.apply(chinese_word_cut)

stop_words_file = "hgd_brand.txt"
stopwords = get_custom_stopwords(stop_words_file)
vect = CountVectorizer(min_df=0,stop_words=frozenset(stopwords),token_pattern='\w+')
term_matrix = pd.DataFrame(vect.fit_transform(x_train.cutted_brand_and_car).toarray(), columns=vect.get_feature_names())

forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
mlp = MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=500)
svm = SVC(kernel='linear',C=1.0,random_state=0)

pipe = make_pipeline(vect, svm)
pipe.fit(x_train.cutted_brand_and_car, y_train)
prediction1 = pipe.predict(x_test.cutted_brand_and_car)

pipe = make_pipeline(vect, forest)
pipe.fit(x_train.cutted_brand_and_car, y_train)
prediction2 = pipe.predict(x_test.cutted_brand_and_car)

pipe = make_pipeline(vect, mlp)
pipe.fit(x_train.cutted_brand_and_car, y_train)
prediction3 = pipe.predict(x_test.cutted_brand_and_car)

prediction_svm = np.array(prediction1)
prediction_forest = np.array(prediction2)
prediction_mlp = np.array(prediction3)
prediction_svm = prediction_svm.tolist()
prediction_forest = prediction_forest.tolist()
prediction_mlp = prediction_mlp.tolist()

li_sign = []

for j in range(len(prediction_svm)):
    if prediction_svm[j] == prediction_forest[j]== prediction_mlp[j]:
        sign = "1"
    else:
        sign = "0"
    print(prediction_svm[j],prediction_forest[j],prediction_mlp[j],"\n",prediction_forest[j]==prediction_svm[j]==prediction_mlp[j],sign)
    li_sign.append(sign)
    
sign = pd.DataFrame(li_sign,columns=['li_sign'])
df1.insert(4,'svm_brand',prediction1)
df1.insert(5,'forest_brand',prediction2)
df1.insert(6,'mlp_brand',prediction3)
df1.insert(7,'sign_brand',sign)
df1.to_csv('ifeng_brand_relationship_pj.csv',mode='a',index=0)

