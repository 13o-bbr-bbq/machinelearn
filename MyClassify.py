#coding:utf-8
import math
import sys
import os
import pickle
import codecs
import numpy
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from MyNaiveBayes import NaiveBayes
from MyMeCab import StringAnalyze

def stems(items):
    stems = StringAnalyze.split_words(items)

    return tuple(stems)

def is_bigger_than_min_tfidf(term, terms, tfidfs):
    if tfidfs[terms.index(term)] > 0.1:
        return True

    return False

def is_smaller_than_min_tfidf(term, terms, tfidfs):
    if tfidfs[terms.index(term)] < 0.1:
        return True

    return False

def tfidf(items):
    #analyzerは文字列を入れると文字列のlistが返る関数

    vectorizer = TfidfVectorizer(analyzer=stems, min_df=1, max_df=50, stop_words='Brillia')
    #corpus = [item for item in items]

    x = vectorizer.fit_transform(items)
    num_samples, num_features = x.shape
    #print(stop_word for stop_word in vectorizer.get_stop_words())
    print("#samples: %d, #features: %d" % (num_samples, num_features))

    # ここから下は返す値と関係ない。tfidfの高い語がどんなものか見てみたかっただけ
    terms = vectorizer.get_feature_names()
    tfidfs = x.toarray()[0]
    print([term for term in terms if is_bigger_than_min_tfidf(term, terms, tfidfs)])
    #print([term for term in terms if is_smaller_than_min_tfidf(term, terms, tfidfs)])

    print('合計%i種類の単語が%iページから見つかりました。' % (len(terms), len(items)))

    # xはtfidf_resultとしてmainで受け取る
    return x, vectorizer

if __name__ == '__main__':
    #訓練済みデータを格納するpklファイルパスを定義
    pkl_nb_path = os.path.join('.\\', 'naive_bayes_classifier.pkl')
    pkl_tfidf_result_path = os.path.join('.\\', 'tfidf_result.pkl')
    pkl_tfidf_vectorizer_path = os.path.join('.\\', 'tfidf_vectorizer.pkl')

    #訓練済みデータ(pkl)が存在する場合、既存の訓練データを使用
    if os.path.exists(pkl_nb_path):
        with open(pkl_nb_path, 'rb') as f:
            nb = pickle.load(f)
    #訓練済みのデータ(pkl)がない場合、学習を行う。
    else:
        nb = NaiveBayes()
        #学習データの読み込み。
        fin = codecs.open('train_data.tsv', 'r', 'utf-8')
        lines = fin.readlines()
        fin.close()

        items = []

        #学習データを一行ずつ学習していく。
        for line in lines:
            words = line[:-2]
            train_words = words.split("\t")
            items.append(train_words[0])
            nb.train(train_words[0], train_words[1])

        #全データの学習が完了したら、訓練済みデータとしてpklファイルに保存する。
        with open(pkl_nb_path, 'wb') as f:
            pickle.dump(nb, f)

        #TF-IDFの計算
        tfidf_result, vectorizer = tfidf(items)

        #TF-IDFの計算はコストが高いため、結果をpklファイルに保存する。
        with open(pkl_tfidf_result_path, 'wb') as f:
            pickle.dump(tfidf_result, f)
        with open(pkl_tfidf_vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    #分類対象の文字列を指定し、学習結果に基づき分類を実施。
    doc = 'カート'
    print('%s => 推定カテゴリ: %s' % (doc, nb.classify(doc)))
