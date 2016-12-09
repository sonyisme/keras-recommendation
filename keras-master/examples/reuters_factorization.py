from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
np.random.seed(1337) # for reproducibility

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
userModel.add(Dense(700, 2))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(itemfea, 1000))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.4))
itemModel.add(TimeDistributedDense(1000, 2))
itemModel.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model=Sequential()
model.add(Cosine([userModel,itemModel])) #should output 2 values 
#model.add(TimeDistributedDense(300, 1))
##model.add(Activation('normalization'))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

for itr in range(0,15):
	print(itr)
	n=3000
	traindoc=np.zeros([n,1000])
	ytrain=np.zeros([n,2])
	trainword=np.zeros([n,2,8982])
	cnt=0
	avg=0
	nb=0
	for d in range(0,8982):
		for w in range(0,1000):
			if cnt==n: # this will ignore last batch
				avg=avg+( model.train_on_batch([traindoc, trainword], ytrain))
				nb=nb+1
				traindoc=np.zeros([n,1000])
				ytrain=np.zeros([n,2])
				trainword=np.zeros([n,2,8982])
				cnt=0
			if X_train[d,w]==1:
				traindoc[cnt]=X_train[d]
				trainword[cnt,0]=xw[w]		
				while True:
					wn=random.randint(0,999)
					if X_train[d,wn]==0:
						break
				trainword[cnt,1]=xw[wn]
				ytrain[cnt,0]=1
				cnt=cnt+1
	print(avg/nb)
	model.save_weights(r'c:\users\t-alie\textfactorization.model.2.'+`itr`)
	#, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
#score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)

model.save_weights(r'c:\users\t-alie\txtfactorization.model.2')