from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
np.random.seed(1337) # for reproducibility
import theano
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

'''
    Train and evaluate a simple MLP on the Reuters newswire topic classification task.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/reuters_mlp.py
    CPU run command:
        python examples/reuters_mlp.py
'''

max_words = 1000
batch_size = 1000
nb_epoch = 15

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print("Vectorizing sequence data...")
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode="binary")
X_test = tokenizer.sequences_to_matrix(X_test, mode="binary")
xw=X_train.transpose()


userfea=1000
itemfea=8982

print("Building model...")
userModel = Sequential()
userModel.add(Dense(userfea, 700))
userModel.add(Activation('tanh'))
userModel.add(Dropout(0.4))
userModel.add(Dense(700, 500))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(itemfea, 1000))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.4))
itemModel.add(TimeDistributedDense(1000, 500))
itemModel.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
itm=itemModel.get_input(train=False)
usr=userModel.get_input(train=False)
itemrep=itemModel.get_output(train=False)
userrep=userModel.get_output(train=False)
model=Sequential()
model.add(Cosine([userModel,itemModel])) #should output 2 values 
#model.add(TimeDistributedDense(300, 1))
##model.add(Activation('normalization'))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))
print("Complie model...")
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Complie outs...")
outv1= theano.function([usr],userrep,allow_input_downcast=True, mode=None)
outv2= theano.function([itm],itemrep,allow_input_downcast=True, mode=None)
print("load W...")
model.load_weights(r'c:\users\t-alie\txtfactorization.model')
print("start predicting...")
df=open(r'c:\users\t-alie\docrep.txt','w')
wf=open(r'c:\users\t-alie\wordrep.txt','w')
for d in range(0,8982):
	dh=userModel.custom_predict([X_train[d]],outv1)
	df.write("%s\n" %dh)

	#, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
#score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
