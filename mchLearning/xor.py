from sklearn import svm
# XOR 계산 결과 data
xor_data = [
    [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]  # [P,Q, 계산 결과 data]
]

# 학습을 위한 data와 정답 label 변수로 분리
data = []
label = []
for row in xor_data: 
    p = row[0] # p = [0, 0, 0]
    q = row[1] # q = [0, 1, 1]
    r = row[2] # r = [1, 0, 1]
    data.append([p, q]) # data = [p,q]
    label.append(r) # label = [r]

# data 학습시키기
clf = svm.SVC() # SVC 객체 생성(3차원 RBF kernel)
clf.fit(data, label) # 주어진 data에 label을 적용시켜, SVM 모델로 학습하기

# data 예측하기
pre = clf.predict(data) # data를 predict하는 pre
print("예측결과:", pre)

# 예측 결과가 맞는지 확인하기
ok = 0
total = 0
for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer:
        ok += 1
    total += 1
print("정답률:", ok, "/", total, "=", ok/total)