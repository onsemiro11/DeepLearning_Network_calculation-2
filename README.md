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


# Conv2D Layers

## Correlation

correlation은 Convolutional Neurons Network 에서 사용하는 대부분의 연산이다.

$$ z = X \otimes F = \sum_{i=1}^{NH-1} \sum_{i=1}^{NW-1} X[i,j]F[i,j] $$
 
[Classical Correlation]

우리가 다루게 되는 것은 convolution 이라 부르지만, 실제로는 correlation이다.
Correlation은 두 신호상이의 유사성을 측정해주는 도구다.
 
  - 실수 값들로 구성된 X 행렬인 입력 데이터가 F(Filter)와 함께 correlation 연산이 된다.

[Correlation with Bias]

 $$ z = X \otimes F = \sum_{i=1}^{NH-1} \sum_{i=1}^{NW-1} X[i,j]F[i,j]+b $$
 
실제로 연산은 Correlation이지만, 주로 Convolution이라고 말을 하는 이유
 
- Convolution : Filter를 180도 돌려서 연산하는 것이다. 어떤 filter를 통과한 output을 계산해준다.
- Correlation : 두 신호 사이의 유사성을 측정해준다.
   > 만약 Filter가 대칭행렬이면 convolution과 correlation이 같아서 옛날부터 같이 불러왔다고 한다.
 

## Correlation and Dot Product

 Dot Product는 Correlation의 Input과 Filter들의 행렬을 Flatten하여 연산하는 것이다.
 
 $$ \sum_{i=1}^{NH-1} \sum_{i=1}^{NW-1} X[i,j]F[i,j]+b = (flatten(x))^T \cdot flatten(F) +b $$
 
Affine Function이라 생각할 수 있지만, Correlation은 데이터 일부를 가져와 연산하는 것이므로, Fully connected layer와 Dense Layer와는 차이점이 있음을 인지해야한다.

## Conv Layer

 
[Window Extraction - Window Formularization]

-사이즈를 정하고, 한칸씩 이동하며 만들어가며 하나하나 window를 구성해간다.
 자동차가 이동하면서 창문으로 보이는 변화되는 모습을 생각해보자. 여기서도 행렬속에서 사이즈에 맞게 한칸씩이동하면서 구성된 값들이 변화된다.
-이 window가 correlation연산에 넣어지게 되는 것이다.
 
[Computations of Conv Layer]

각 window와 kernal을 correlation해준 후, bias를 더해주면 Z값이 생성된다.

(affine Function이라 생각할 수 있지만, 변화되는 입력 데이터라는 점이 다르다.)

생성된 Z 값은 activation Function에 넣어 나온 출력값이 한 모듈을 통과한 값이다.
 
[n-Channel Input]

Data가 이미지라면, R G B의 각 window들이 형성되어야할 것이다. 만약 이미지 속 그림이 5X5크기라면, f = 5  nc = 5
-> window는 실수 5X5X3이 될것 이다.

[Conv Layers]

입력데이터 또는 전 layer의 출력값이 kernal를 통가하면 필터의 개수(ℓ1...)로 변한다. 
 
nH 와 nW도 각 kernal에 의해 변하게 된다.

filter는 window크기와 동일 하기 때문에
nH 와 nW를 window로 묶어 나눠진 후에, filter와 correlation 연산이 진행된다는 점을 생각하면

- nH[2] = nH[1] - f[2] +1
- nW[2] = nW[1]- f[2] +1

의 shape로 변화된다.
