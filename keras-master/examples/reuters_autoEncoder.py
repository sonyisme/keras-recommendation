from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
np.random.seed(1337) # for reproducibility
from keras.layers import containers

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul,AutoEncoder
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
encoder = containers.Sequential([Dense(1000, 700), Dense(700, 500)])
decoder = containers.Sequential([Dense(500, 700), Dense(700, 1000)])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(X_train, X_train, nb_epoch=15, batch_size=1024, verbose=1, show_accuracy=True, validation_split=0.1)
df=open(r'f:\autoencoderrep.txt')
dh= model.predict(X_train)
for doc in dh:
	for v in doc:
		df.write("%s " %v)
	df.write("\n")
df.close()
	#, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
#score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)

model.save_weights(r'c:\users\t-alie\txtfactorization.Auto.model')