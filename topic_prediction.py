
# coding: utf-8

# In[ ]:


# 预测主题
import os
import jieba
import gensim
import numpy as np
import jieba.posseg
from gensim import corpora, models
import re
import pandas as pd


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf8').readlines()]
    return stopwords

pos = ['n', 'eng', 'ns', 'x', 'nr', 'j', 'a', 'nt', 'l', 'b', 'nz', 'nrt', 'i']
jieba.load_userdict('topic_dictionary.txt')
stopwords = stopwordslist('topic_stopwords.txt')


def test_doc_bydate(corpus_path_test):
    test_doc = []
    file_list_test = os.listdir(corpus_path_test)
    for file_path in file_list_test:
        full_name_test = corpus_path_test + file_path
        file_read = open(full_name_test, "r", encoding="utf8")
        content_test = file_read.read()
        content_seg_test = jieba.posseg.cut(content_test)
        content_seg_li_test = []
        for i in content_seg_test:
            if i.flag in pos:
                content_seg_li_test.append(i.word)
        content_seg_withoutsingle_test = []
        for item in content_seg_li_test:
            if len(str(item)) > 1:
                content_seg_withoutsingle_test.append(item)
        test_doc.append(
            [item for item in content_seg_withoutsingle_test if str(item) not in stopwords])
    return test_doc


def test_doc_accumulated(corpus_path_test):
    catelist_test = os.listdir(corpus_path_test)
    test_doc = []
    for mydir in catelist_test:
        class_path_test = corpus_path_test + mydir + '/'
        file_list_test = os.listdir(class_path_test)
        for file_path in file_list_test:
            full_name_test = class_path_test + file_path
            file_read = open(full_name_test, "r", encoding="utf8")
            content_test = file_read.read()
            content_seg_test = jieba.posseg.cut(content_test)
            content_seg_li_test = []
            for i in content_seg_test:
                if i.flag in pos:
                    content_seg_li_test.append(i.word)
            content_seg_withoutsingle_test = []
            for item in content_seg_li_test:
                if len(str(item)) > 1:
                    content_seg_withoutsingle_test.append(item)
            test_doc.append(
                [item for item in content_seg_withoutsingle_test if str(item) not in stopwords])
    return test_doc


def topic_prediction(test_doc):
    n = 1
    number_topic1 = 0
    number_topic2 = 0
    number_topic3 = 0
    number_topic4 = 0
    number_topic5 = 0
    number_topic6 = 0
    lda = models.ldamodel.LdaModel.load('lda_collectivity.model')
    dic = corpora.Dictionary.load('dic_collectivity.dict')
    for text in test_doc:
        doc_bow = dic.doc2bow(text)
        doc_lda = lda[doc_bow]
        # print(doc_lda)
        probability = []
        for topic in doc_lda:
            #print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))
            probability.append(topic[1])
        which_one = probability.index(max(probability)) + 1
        #print("第几篇：" , n)
        #print("属于哪个主题：" , which_one , "\n")
        if which_one == 1:
            print()
            number_topic1 += 1
        elif which_one == 2:
            number_topic2 += 1
        elif which_one == 3:
            number_topic3 += 1
        elif which_one == 4:
            number_topic4 += 1
        elif which_one == 5:
            number_topic5 += 1
        elif which_one == 6:
            number_topic6 += 1
        n += 1

    print("主题1篇数：", number_topic1, "\n")
    print("主题2篇数：", number_topic2, "\n")
    print("主题3篇数：", number_topic3, "\n")
    print("主题4篇数：", number_topic4, "\n")
    print("主题5篇数：", number_topic5, "\n")
    print("主题6篇数：", number_topic6, "\n")

if __name__ == "__main__":
    base_path = "news_data/by_brand/"
    dirNameLis = os.listdir(base_path)
    fileLis_bydate = [base_path + i + "/20180828/" for i in dirNameLis]
    path_zongti_bydate = "news_data/by_date/20180828/"
    path_cdcz_before_bydate = "news_data/by_chengdu/exhibition_before/20180828/"
    path_cdcz_intermediate_bydate = "news_data/by_chengdu/exhibition_intermediate/20180906/"
    path_cdcz_after_bydate = "news_data/by_chengdu/exhibition_after/20180910/"
    fileLis_accumulated = [base_path + i + "/" for i in dirNameLis]
    path_zongti_accumulated = "news_data/by_date/"
    path_cdcz_before_accumulated = "news_data/by_chengdu/exhibition_before/"
    path_cdcz_intermediate_accumulated = "news_data/by_chengdu/exhibition_intermediate/"
    path_cdcz_after_accumulated = "news_data/by_chengdu/exhibition_after/"

    for i in fileLis_bydate:
        print(
            "8月28日" + re.match("news_data/by_brand/([^/]*)/20180828/", i).group(1) + "主题如下：")
        topic_prediction(test_doc_bydate(i))
    print("8月28日总体主题如下：")
    topic_prediction(test_doc_bydate(path_zongti_bydate))
    print("8月28日成都车展主题如下：")
    topic_prediction(test_doc_bydate(path_cdcz_before_bydate))
    for i in fileLis_accumulated:
        print(
            re.match("news_data/by_brand/([^/]*)/", i).group(1) + "主题如下（按累积）：")
        topic_prediction(test_doc_accumulated(i))
    print("总体主题如下（按累积）：")
    topic_prediction(test_doc_accumulated(path_zongti_accumulated))
    print("成都车展展前主题如下（按累积）：")
    topic_prediction(test_doc_accumulated(path_cdcz_before_accumulated))
    print("成都车展展中主题如下（按累积）：")
    topic_prediction(test_doc_accumulated(path_cdcz_intermediate_accumulated))
    print("成都车展展后主题如下（按累积）：")
    topic_prediction(test_doc_accumulated(path_cdcz_after_accumulated))
