import pandas as pd
import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt

# アップロードされたファイル名を取得
file_name = "/Users/chinenryoya/Downloads/wine+quality/winequality-white.csv"

# CSVファイルをデータフレームとして読み込む
data = pd.read_csv(file_name, delimiter=';')  # デリミタがセミコロンで区切られている場合

# 列名の表示
print(data.columns)

# 特徴量とターゲットの分離（適切な列名を使用）
X = data.drop("quality", axis=1)  # 正しいターゲット列名を使用
y = data['quality']  # 正しいターゲット列名を使用

# UMAPの適用
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

# 結果のデータフレーム化
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
embedding_df['Quality'] = y

# 可視化
plt.figure(figsize=(12, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Quality', palette='viridis', data=embedding_df, legend='full')
plt.title('UMAP projection of the Wine Quality dataset')
plt.show()

# 相関係数の計算
correlations = X.corrwith(y)

# 相関係数の表示
print(correlations)

# バーグラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.index, y=correlations.values)
plt.title('Correlation of Features with Wine Quality')
plt.xlabel('Features')
plt.ylabel('Correlation with Quality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
