#coding:utf-8
import math
import sys
import os
import MeCab

class StringAnalyze:
    def split_words(sentence):
        tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
        mecab_result = tagger.parse(sentence)
        info_of_words = mecab_result.split('\n')
        words = []

        for info in info_of_words:
            if info == 'EOS' or info == '':
                break

            info_elems = info.split(',')
            elems = info_elems[0].split('\t')
            words.append(elems[0])

        #return tuple(words)
        return words
