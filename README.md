# Loss Function

모델의 output 즉, 예측값과 실제 데이터 셋과의 차이를 수치화시키는 함수다.(이 차이를 출이는 것이 목표이며, loss값이 필요한 이유다.)

어떤 종류의 Loss Function이든, 예측값과 실제값이 비슷할수록 0 & 비슷하지 않을수록 1 또는 $\infty$ 에 가까워야한다는 개념으로부터 파생된다는 것을 인지하자.

# MSE(Mean Squared Error)

J = $1 \over n$ $\sum$ ( y - $\hat{y}$ ) $^2$

이 또한, 둘이 비슷할수록 0에 가까워지고, 둘의 차이가 클수록 $\infty$ 로 발산한다.

$\hat{y}$ 은 $\hat{y}$ = wx + b 이다. (w는 weightm, b는 bias)

( y - $\hat{y}$ ) $^2$ 는 아래로 볼록한 모형의 그래프를 형성할 것이다.

보통 regression에서 주로 사용된다.

# BCE(Binary Cross Entropy)

Binary Classification을 할 때 사용하는 Loss Function이다.

![Pasted Graphic](https://user-images.githubusercontent.com/49609175/210971610-3b0b8e23-a1ce-4aad-987c-abaaa0caf924.png)
