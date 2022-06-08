from icrawler.builtin import BingImageCrawler
import glob
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


#クワガタの画像を収集
crawler = BingImageCrawler(storage={"root_dir":"images"}) #ダウンロード先のディレクトリを指定
crawler.crawl(keyword="クワガタ", max_num=50) #クロール実行

#カマキリの画像を収集
crawler = BingImageCrawler(storage={"root_dir":"images2"}) #ダウンロード先のディレクトリを指定
crawler.crawl(keyword="カマキリ", max_num=50) #クロール実行

#全ての画像ファイルのパスを取得する
files1 = glob.glob("images/*.jpg")
files2 = glob.glob("images2/*.jpg")

files1[len(files1):len(files2)] = files2

#画像データを格納するりすと
image_list = []

#ファイルパスから画像を読み込み
for imgpath in files1:
  image = cv2.imread(imgpath) #画像を読み込み
  image = cv2.resize(image, (60, 60)) #画像のサイズを変更

  #正規化
  image = image / 255 #[0~1]にスケーリング

  #画像をリストに追加
  image_list.append(image)

#ラベルを作成する
label_list = []
label_list2 = []

#50ずつ作成
for i in range(50):
    label_list.append("stag_beetle")

for i in range(50):
    label_list2.append("mantis")

#ラベルを統合
label_list[len(label_list):len(label_list2)] = label_list2

#学習データとテストデータを作成
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2) #20件をテストデータ、それ以外を学習データに分ける


#テーブルデータを作成
columns = ['type'] #列名を指定
df_train = pd.DataFrame(y_train,  columns=columns)
df_test = pd.DataFrame(y_test, columns=columns)
print("--TableData--")
print("train df = \n", df_train)
print("test df = \n", df_test)

# 文字列（カテゴリ変数）をダミー変数に変換
y_train = pd.get_dummies(df_train)
y_test = pd.get_dummies(df_test)
print("--dummies--")
print("train dummies = \n", y_train)
print("test dummies = \n", y_test)

#リスト型を配列型に変換
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(80, 10800) #60×60×3（RGB）
x_test = x_test.reshape(20, 10800)

#層を定義
model = models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(10800,))) #activationは活性化関数
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax')) #第1引数分類させる数

#モデルを構築
model.compile(optimizer=tf.optimizers.Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

#EaelyStoppingの設定
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)

#モデルを学習させる
log = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=True, validation_data=(x_test, y_test), callbacks=[early_stopping])

#モデルを評価する
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('損失:', test_loss)
print('評価:', test_accuracy)