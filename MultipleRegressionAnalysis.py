# -*- coding: utf-8 -*-
import statsmodels.api as sm
import pandas as pd
from matplotlib import pyplot as plt

def main():
    # 過去の案件データの取り込み(診断工数、画面数、診断員の経験値、Webサーバのレスポンス速度)
    data = pd.read_csv('pen_manhour_single.csv', skiprows=1, names=['man_hour', 'target_num', 'exp', 'speed'], encoding='UTF_8')

    # 画面数、診断員の経験値、Webサーバのレスポンス速度を説明変数として定義
    x = data[['target_num', 'exp', 'speed']]

    # 定数項をxに加える
    x = sm.add_constant(x)

    # 診断工数を目的変数yとして定義
    y = data['man_hour']

    # モデルを定義
    model = sm.OLS(y, x, prepend=False)

    # 重回帰分析の実行
    results = model.fit()

    # 分析結果の表示
    print(results.summary())

if __name__ == '__main__':
    main()
