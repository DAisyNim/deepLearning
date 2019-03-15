import pandas as pd
from sklearn import svm, metrics

# XOR 연산
xor_input = [
    [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]  # [P,Q, 계산 결과 data]
]

# input을 학습 data와 test 전용 data로 나누기
xor_df = pd.DataFrame(xor_input)  # pandas의 dataframe 으로 , 학습 data와 label 쉽게 구분
xor_data = xor_df.ix[:, 0:1]  # data
xor_label = xor_df.ix[:, 2]  # label

# data 학습 & 예측
clf = svm.SVC()   # SVC 객체 생성(3차원 RBF kernel)
clf.fit(xor_data, xor_label)  # 주어진 data에 label을 적용시켜, SVM 모델로 학습하기
pre = clf.predict(xor_data)  # data를 predict하는 pre

# 정답률 구하기
ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률=", ac_score)
