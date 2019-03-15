# perceptron: input data(xi)를 activated function에 대입한 값 (xi*wi의 합)인 실제 data와
#           예측 data를 비교하여, 두 값이 다른 경우 오차를 줄이는 방향으로 wi(가중치)를 업데이트 하는 것.
# wi(p) = wi(p) + a * xi(p)* e(p) // 이때 a는 학습률, xi(p)는 input data, e(p)는 오차
# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220948258166&categoryNo=87&parentCategoryNo=0&viewDate=&currentPage=4&postListTopCurrentPage=1&from=postList&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=4

import numpy as np 
class Perceptron():
    def __init__(self, thresholds= 0.0, eta =0.01, n_iter=10): 
        self.thresholds=thresholds # 임계값
        self.eta=eta # learning rate(학습률)
        self.n_iter=n_iter # 학습 횟수 

    def fit(self,X,y): # X는 실제 data, y는 예측 data
        self.w_ = np.zeros(1+X.shape[1]) # 가중치를 numpy의 배열로 정의
                                         # X.shape[1]은 input data의 개수를 의미함
        self.errors_=[] # 반복 회수에 따라 실제 data와 예측 data가 다른 횟수(오류 횟수)를 저장하는 변수

        for _ in range(self.n_iter):  # 학습횟수만큼 반복함 
            errors = 0      # 초기 오류 횟수는 0
            for xi, target in zip(X,y): 
                update=self.eta*(target-self.predict(xi)) # 학습률 * 오차
                self.w_[1:] += update*xi # 학습률 * 오차 * input data + 기존 가중치 
                self.w_[0] += update 
                errors += int(update!=0.0) # 만약 오차가 0이라면, update도 0이고, 그 때 error++
            self.errors_.append(errors)
            print(self.w_)
        return self

    def net_input(self,X):
        return np.dot(X, self.w_[1:])+self.w_[0] # wi * xi 들의 합 

    def predict(self, X):
        return np.where(self.net_input(X) > self.thresholds, 1, -1) 
        # net_input이 threshold보다 크면 1, 아니면 -1 
        # activated function을 나타낸 함수 