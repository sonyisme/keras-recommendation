# Keras Implementation of Recommender Systems

This library contains a modified version of Keras (mostly in the layers/core.py) to implement various recommender systems, including the Deep Structured Semantic Model (DSSM), Multi-View DSSM (MV-DSSM), Temporal DSSM (TDSSM) and matrix factorization (MF).

The examples can be found in the examples/ folder.

Contact: Yang Song (sonyisme AT google dot com)

Homepage: http://sonyis.me

## Examples

### Temporal DSSM
```python
staticmodel=Sequential()
staticmodel.add(Dense(300,300))
staticmodel.add(Activation('tanh'))
tempmodel=Sequential()
tempmodel.add(LSTM(tempFea,300))
model=Sequential()
model.add(Merge([staticmodel, tempmodel],mode='concat'))
model.add(Dense(300+300,300))
model.add(Activation('tanh'))
```

### DSSM and Multi-view DSSM
```python
userModel = Sequential()
userModel.add(Dense(3883, 300))
userModel.add(Activation('tanh'))
userModel.add(Dropout(0.4))
userModel.add(Dense(300, 300))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(6039, 300))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.4))
itemModel.add(TimeDistributedDense(300, 300))
itemModel.add(Activation('tanh'))

model=Sequential()
model.add(Cosine([userModel,itemModel])) #should output 2 values
model.add(Reshape(2))
```
### Matrix Factorization on MovieLens
```python
serModel = Sequential()
userModel.add(Dense(1682, 500))
userModel.add(Activation('tanh'))
userModel.add(Dropout(0.4))
userModel.add(Dense(500, 500))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(943, 500))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.4))
itemModel.add(TimeDistributedDense(500, 500))
itemModel.add(Activation('tanh'))
model=Sequential()
model.add(ElementMul([userModel,itemModel])) #should output 2 values
model.add(TimeDistributedDense(500, 1))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))
```

## References
[1] Yang Song, Ali Elkahky, and Xiaodong He, Multi-Rate Deep Learning for Temporal Recommendation, in SIGIR 2016.

[2] Ali Mamdouh Elkahky, Yang Song, and Xiaodong He, A Multi-View Deep Learning Approach for User Modeling in Recommendation Systems, in WWW 2015.

[3] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero and Larry Heck, Learning Deep Structured Semantic Models for Web Search using Clickthrough Data, in CIKM 2013.

