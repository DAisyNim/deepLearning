# Iris sanguinea 구분하기 예제
# pandas의 test data 중 Iris data를 이용
# https://github.com/pandas-dev/pandas/blob/master/pandas/tests/data/iris.csv

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# pandas를 이용해 iris.csv 파일 읽어들이기 (header제거, 줄바꿈 제거 등등 한번에 해결)
csv = pd.read_csv('iris.csv', header=None, error_bad_lines=False)

# 열 이름을 이용하여 분할 : 필요한 열 추출하기
csv_data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
csv_label = csv["Name"]

# 학습 전용 data와 test data를 분할
train_data, test_data, train_label, test_label = \
    train_test_split(csv_data, csv_label)

# data 학습시키고 예측하기
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 정답률 구하기
ac_score = metrics.accuracy_score(test_label, pre)
print("정답률 =", ac_score)
