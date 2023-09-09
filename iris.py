# 基本ライブラリ
from typing_extensions import dataclass_transform
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# データセット読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値の列を追加
df["target"] = iris.target

# 「教師あり学習：分類 1」の講義を参照
# iris.feature_names : sepal length(cm)/sepal width(cm)/petal length(cm)/petal width(cm)の4つの名前がそのまま格納されている
# iris.data : 上記 4つの説明変数の数値データ
# iris.target : 目的変数のカテゴリー分類データ(0~3)

# 目標値を数値データから花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

# 予測モデルの構築
x = iris.data[:, [0, 2]]   # 全ての行、0列目(sepal length)と2列目(petal length)をスライスで選択
y = iris.target

# ロジスティック回帰モデル
model = LogisticRegression()
model.fit(x, y)

# サイドバーの入力画面
st.sidebar.header("Input features")

# スライダー
sepalValue = st.sidebar.slider("sepal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider("petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame([],columns=['data','sepal length (cm)','petal length (cm)'])
record = pd.Series(['data',sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data',inplace=True)

# 入力値の値を表示
st.write(value_df)

# 予測値のデータフレーム
pred_probs = model.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')

