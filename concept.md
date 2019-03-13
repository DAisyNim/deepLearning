
# machine learning: 

기계에게 데이터를 학습시키고, 데이터에 포함된 규칙성 or 패턴을 발견하게 하는 처리

  1) 기계에게 학습할 data를 input한다. 
  2) 사람이 특징량(이런 data의 특징을 수치화한 것)을 정의한다.
  3) 특징량을 기반으로 규칙과 패턴을 찾아 학습시킴
  
 ## 종류 
 
  - Supervised Learning (주로 예측)
    - 정답 data (목적 variable)를 input하면, 그것을 제외한 input을 기반으로 출력 결과가 최대한 정답 data에 가까워지도록 특징량을 추출하여 모델 구축
    - 훈련 데이터(Training Data)로부터 주어진 데이터에 대해 예측하고자 하는 값을 올바로 추측해내는 하나의 함수를 유추. 
    - decision tree, random forest, regression, logistic regression, artificial neural network(ANN), deep learning ...
    * 예시 : 고장날 기계 예측
    
  - Unsupervised Learning (주로 지식 발견)
    - 데이터가 어떻게 구성되었는지
    - 정답 data (목적 variable)가 포함되지 않음
    - coresspondense analysis, association analysis, network analysis, Principal component analysis(PCA) ...
    * 예시: clustering, 고객을 그룹으로 나누기
    
  
  
  
  
  
  
  
  
 ## Neural Network


![neural_network](./img/neural_network.png)

 (사진 출처 : https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5_%EC%8B%A0%EA%B2%BD%EB%A7%9D)
 
    레이어의 종류는 input, hidden, output 이고  
    각 레이어에는 O 이라고 표현되는 node, - 라고 표현되는 edge(link)가 있으며 node를 연결한다.
  
 
 
 ### 종류
 

 *  artificial neural network, ANN
    - deep learning의 기반이 되는 기술로, 인간 뇌 신경 세포를 모방해 만든 수학적 모델.
 *  deep neural network, DNN
    - 모든 node가 결합한 전결합 신경망(Fully-connected Neural Network, FNN)
     -> 딥러닝의 기본적인 형태
    - 합성곱 레이어, 풀링 레이어가 추가된 합성곱신경망(CNN, Convolutional Neural Network)
     -> 피사체 인식에 많이 쓰임
    - 재귀 신경망(recursive neural network)
     -> 텍스트나 음성 data에 
  
  
  
