
# coding: utf-8

# In[ ]:


# 最终的车系分类器
import pandas as pd
import numpy as np
import jieba
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

df1 = pd.read_csv('ifeng_brand_relationship_pj.csv', encoding='gb18030')
df2 = pd.read_csv('Test.csv', encoding='utf8')
rs = pd.DataFrame(columns=['brand_name', 'brand_id', 'car_name', 'car_id', 'factory', 'svm_brand',
                           'forest_brand', 'mlp_brand', 'sign_brand', 'standard_brand', 'svm_car', 'forest_car', 'mlp_car'])


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
names = df1.standard_brand
names = names.dropna()
names = np.array(names)
names = names.tolist()
names = sorted(set(names), key=names.index)

for name in names:
    df_car_test = df1.loc[df1['standard_brand'] == name]
    df_car_train = df2.loc[df2['standard_brand'] == name]
    x_test = df_car_test[['car_name']]
    x_train = df_car_train[['car_name']]
    y_train = df_car_train.standard_car
    x_train['cutted_car_name'] = x_train.car_name.apply(chinese_word_cut)
    x_test['cutted_car_name'] = x_test.car_name.apply(chinese_word_cut)

    stop_words_file = "hgd_car.txt"
    stopwords = get_custom_stopwords(stop_words_file)
    vect = CountVectorizer(min_df=0, stop_words=frozenset(
        stopwords), token_pattern='\w+')
    term_matrix = pd.DataFrame(vect.fit_transform(
        x_train.cutted_car_name).toarray(), columns=vect.get_feature_names())

    forest = RandomForestClassifier(
        criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    mlp = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter=500)
    svm = SVC(kernel='linear', C=1.0, random_state=0)

    pipe = make_pipeline(vect, svm)
    pipe.fit(x_train.cutted_car_name, y_train)
    prediction1 = pipe.predict(x_test.cutted_car_name)

    pipe = make_pipeline(vect, forest)
    pipe.fit(x_train.cutted_car_name, y_train)
    prediction2 = pipe.predict(x_test.cutted_car_name)

    pipe = make_pipeline(vect, mlp)
    pipe.fit(x_train.cutted_car_name, y_train)
    prediction3 = pipe.predict(x_test.cutted_car_name)

    df_car_test.insert(9, 'svm_car', prediction1)
    df_car_test.insert(10, 'forest_car', prediction2)
    df_car_test.insert(11, 'mlp_car', prediction3)
    rs = pd.concat([rs, df_car_test])

prediction_svm = np.array(rs.svm_car)
prediction_forest = np.array(rs.forest_car)
prediction_mlp = np.array(rs.mlp_car)
prediction_svm = prediction_svm.tolist()
prediction_forest = prediction_forest.tolist()
prediction_mlp = prediction_mlp.tolist()
li_sign_car = []

for j in range(len(prediction_svm)):
    if prediction_svm[j] == prediction_forest[j] == prediction_mlp[j]:
        sign_car = "1"
    else:
        sign_car = "0"
    li_sign_car.append(sign_car)

sign_car = pd.DataFrame(li_sign_car, columns=['li_sign_car'])
rs.insert(12, 'sign_car', sign_car)
rs.to_csv('pcauto_car_relationship.csv', mode='a', index=0)
