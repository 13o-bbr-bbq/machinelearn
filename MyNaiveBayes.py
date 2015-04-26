#coding:utf-8
import math
import sys
import os
from MyMeCab import StringAnalyze

class NaiveBayes:
    def __init__(self):
        self.vocabularies = set()
        self.word_count = {}  # {'花粉症対策': {'スギ花粉': 4, '薬': 2,...} }
        self.category_count = {}  # {'花粉症対策': 16, ...}

    def word_count_up(self, word, category):
        self.word_count.setdefault(category, {})
        self.word_count[category].setdefault(word, 0)
        self.word_count[category][word] += 1
        self.vocabularies.add(word)

    def category_count_up(self, category):
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    def train(self, doc, category):
        words = StringAnalyze.split_words(doc)
        for word in words:
            self.word_count_up(word, category)
        self.category_count_up(category)

    def prior_prob(self, category):
        num_of_categories = sum(self.category_count.values())
        num_of_docs_of_the_category = self.category_count[category]

        return num_of_docs_of_the_category / num_of_categories

    def num_of_appearance(self, word, category):
        if word in self.word_count[category]:
            return self.word_count[category][word]

        return 0

    def word_prob(self, word, category):
        # ベイズの法則の計算
        numerator = self.num_of_appearance(word, category) + 1  # +1は加算スムージングのラプラス法
        denominator = sum(self.word_count[category].values()) + len(self.vocabularies)

        # Python3では、割り算は自動的にfloatになる
        prob = numerator / denominator

        return prob

    def score(self, words, category):
        score = math.log(self.prior_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))

        return score

    def classify(self, doc):
        best_guessed_category = None
        max_prob_before = -sys.maxsize
        words = StringAnalyze.split_words(doc)

        for category in self.category_count.keys():
            prob = self.score(words, category)
            if prob > max_prob_before:
                max_prob_before = prob
                best_guessed_category = category

        return best_guessed_category
