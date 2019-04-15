
# coding: utf-8

# In[ ]:


# 计算主题名称
import pandas as pd
import gensim
from gensim import models
df = pd.read_csv('news_data/weight_matrix2.csv', encoding='gb18030')
df_weight_label1 = df[["topic_word_total8", "新车上市"]]
df_weight_label2 = df[["topic_word_total8", "品牌战略"]]
df_weight_label3 = df[["topic_word_total8", "产品规划"]]
df_weight_label4 = df[["topic_word_total8", "车型测试"]]
df_weight_label5 = df[["topic_word_total8", "消费者口碑营销"]]
df_weight_label6 = df[["topic_word_total8", "产品性能"]]


def get_weight_eachlabel(df_weight_label):
    dict_weight_bylabel = df_weight_label.set_index(
        'topic_word_total8').T.to_dict('list')
    return dict_weight_bylabel

dict_weight_label1 = get_weight_eachlabel(df_weight_label1)
dict_weight_label2 = get_weight_eachlabel(df_weight_label2)
dict_weight_label3 = get_weight_eachlabel(df_weight_label3)
dict_weight_label4 = get_weight_eachlabel(df_weight_label4)
dict_weight_label5 = get_weight_eachlabel(df_weight_label5)
dict_weight_label6 = get_weight_eachlabel(df_weight_label6)

lda = models.ldamodel.LdaModel.load('lda_collectivity.model')


def get_topicword_and_importance(i):
    importance = []
    topic_word = []
    for k, v in lda.show_topic(i, 200):
        importance.append(v)
        topic_word.append(k)
    return importance, topic_word

importance_topic1, topic_word_topic1 = get_topicword_and_importance(0)
importance_topic2, topic_word_topic2 = get_topicword_and_importance(1)
importance_topic3, topic_word_topic3 = get_topicword_and_importance(2)
importance_topic4, topic_word_topic4 = get_topicword_and_importance(3)
importance_topic5, topic_word_topic5 = get_topicword_and_importance(4)
importance_topic6, topic_word_topic6 = get_topicword_and_importance(5)


def get_topic_name(topic_word_bytopic, importance_bytopic):
    prob_topic = []
    probability_label1 = 0
    probability_label2 = 0
    probability_label3 = 0
    probability_label4 = 0
    probability_label5 = 0
    probability_label6 = 0
    for index, value in enumerate(topic_word_bytopic):
        try:
            a = dict_weight_label1[value]
        except:
            a = dict_weight_label1["MARVEL"]
        try:
            b = dict_weight_label2[value]
        except:
            b = dict_weight_label2["MARVEL"]
        try:
            c = dict_weight_label3[value]
        except:
            c = dict_weight_label3["MARVEL"]
        try:
            d = dict_weight_label4[value]
        except:
            d = dict_weight_label4["MARVEL"]
        try:
            e = dict_weight_label5[value]
        except:
            e = dict_weight_label5["MARVEL"]
        try:
            f = dict_weight_label6[value]
        except:
            f = dict_weight_label6["MARVEL"]
        probability_label1 += importance_bytopic[index] * a[0]
        probability_label2 += importance_bytopic[index] * b[0]
        probability_label3 += importance_bytopic[index] * c[0]
        probability_label4 += importance_bytopic[index] * d[0]
        probability_label5 += importance_bytopic[index] * e[0]
        probability_label6 += importance_bytopic[index] * f[0]

    prob_topic.append(probability_label1)
    prob_topic.append(probability_label2)
    prob_topic.append(probability_label3)
    prob_topic.append(probability_label4)
    prob_topic.append(probability_label5)
    prob_topic.append(probability_label6)
    which_label = prob_topic.index(max(prob_topic)) + 1
    print(prob_topic)
    return which_label


def show_topicname():
    label_of_topic1 = get_topic_name(topic_word_topic1, importance_topic1)
    label_of_topic2 = get_topic_name(topic_word_topic2, importance_topic2)
    label_of_topic3 = get_topic_name(topic_word_topic3, importance_topic3)
    label_of_topic4 = get_topic_name(topic_word_topic4, importance_topic4)
    label_of_topic5 = get_topic_name(topic_word_topic5, importance_topic5)
    label_of_topic6 = get_topic_name(topic_word_topic6, importance_topic6)
    print("主题1名称：", label_of_topic1)
    print("主题2名称：", label_of_topic2)
    print("主题3名称：", label_of_topic3)
    print("主题4名称：", label_of_topic4)
    print("主题5名称：", label_of_topic5)
    print("主题6名称：", label_of_topic6)


if __name__ == "__main__":
    show_topicname()
