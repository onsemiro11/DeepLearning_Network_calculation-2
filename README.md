# Loss Function

모델의 output 즉, 예측값과 실제 데이터 셋과의 차이를 수치화시키는 함수다.(이 차이를 출이는 것이 목표이며, loss값이 필요한 이유다.)

어떤 종류의 Loss Function이든, 예측값과 실제값이 비슷할수록 0 & 비슷하지 않을수록 1 또는 $\infty$ 에 가까워야한다는 개념으로부터 파생된다는 것을 인지하자.

# MSE(Mean Squared Error)

$$MSE = \frac{1}{N} \sum_{i=1}^{N} ( y - \hat{y} ) ^2$$

이 또한, 둘이 비슷할수록 0에 가까워지고, 둘의 차이가 클수록 $\infty$ 로 발산한다.

$\hat{y}$ 은 $\hat{y}$ = wx + b 이다. (w는 weight, b는 bias)

( y - $\hat{y}$ ) $^2$ 는 아래로 볼록한 모형의 그래프를 형성할 것이다.

보통 regression에서 주로 사용된다.

# BCE(Binary Cross Entropy)

Binary Classification을 할 때 사용하는 Loss Function이다.

$$BCE = H( y , \hat{y} ) = - \frac{1}{N} [ y \log ( \hat{y} ) + ( 1 - y ) \log ( 1 - \hat{y} )] $$

y = 0이면, 앞에 항이 0으로 사라져서 뒤에 항만 계산 가능하고 / y = 1이면, 뒤에 항이 0으로 사라져서 앞에 항만 계산 가능하다.

 > Binary Classification의 특성상 y값이 0 또는 1이 나오기 때문에 위와 같은 식을 구현하여 Loss 값을 추출할 수 있게 됨!

아래 그래프를 확인하면, 더 쉽게 식을 이해할 수 있다.

![Pasted Graphic](https://user-images.githubusercontent.com/49609175/210971610-3b0b8e23-a1ce-4aad-987c-abaaa0caf924.png)

# CCE(Categorical Cross Entropy)

클래스가 3개 이상인 데이터를 대상으로 사용하는 Loss Function이다. 주로 Softmax함수를 Activation Function으로 사용한다.

출력층 노드 수는 클래스 수와 동일해야하고 동일하다. 출력된 벡터는 각 클래스에 속할 확률이 나오며, 총합이 1이다.

$$CCE = -\frac{1}{N} \sum_{i=1}^{N} \sum_{i=1}^{C} t_{ij}\log(y_{ij})$$

데이터셋 수 N개 만큼 합하여 평균을 낸 것이다.

여기서 모든 클래스 수에 맞는 one hot encoding을 하게 되면 무의미한 0 달이 생겨 저장공간 이슈가 나타난다.

이런 0을 없애서 matrix 상태를 하나의 vector 상태로 변환해준다. $\Rightarrow$ SCCE(Sparse Categroical Cross Entropy)


# conv Layers
