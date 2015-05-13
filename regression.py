# -*- coding: utf-8 -*-
import statsmodels.api as sm
import pandas as pd

def main():
    # 分析対象データの読み込み
    data = pd.read_csv('sample.csv', skiprows=1, names=['height', 'weight'], encoding='UTF_8')

    # 身長を説明変数として定義
    x = data[['height']]

    # 定数項をxに加える
    x = sm.add_constant(x)

    # 体重を目的変数として定義
    y = data[['weight']]

    # モデルを定義
    model = sm.OLS(y, x, prepend=False)

    # 単回帰分析の実行
    results = model.fit()

    # 分析結果の表示
    print(results.summary())

if __name__ == '__main__':
    main()
