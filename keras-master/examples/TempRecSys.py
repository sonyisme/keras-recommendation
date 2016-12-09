from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from itertools import repeat
import csv
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

#input is 4*embeddingdim 
#output is embeddingdim
def loadData(file=r'c:\f\f\cortana_lstm_new_same_length.txt', num_features=61):
    x_train = []
    y_train = []
    current_x = []
    current_y = []
    current_index = 1
    with open(file, 'rb') as input_file:
        input = csv.reader(input_file, delimiter='\t')
        for row in input:
            print(row[0])
            if int(row[0])!=current_index:
                # a new training instance
                print("new")
                x_train.append(current_x)
                y_train.append(current_y)
                current_index=int(row[0])
                current_x = []
                current_y = []
            else:
                print("old")
            current_x.append(map(int, row[2:2+num_features]))
            current_y.append(int(row[1]))
    # don't forget the last instance
    x_train.append(current_x)
    y_train.append((current_y))
    return x_train, y_train



if __name__ == "__main__":
	numfeatures = 61 # the input feature size
	youtput = 1 # the output prediction (only predict 0 or 1)
	embeddingdim=300
	x_train, y_train= loadData(num_features=numfeatures)
	print(len(x_train), 'train sequences')	
	#print(x_train[0])
	#print(x_train[1])
	# padding sequence
	print("padding sequences")
	#x_train = sequence.pad_sequences(x_train,maxlen= 20)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	print(x_train.shape, ' x_train shape')
	print(y_train.shape, ' y_train shape')
	print(y_train)
	print('Build model...')
	
	model_user_Item_globalItem = Sequential()
	model_user_Item_globalItem.add(Dense(embeddingdim*3, embeddingdim))
	model_temp_userItem = Sequential()
	model_temp_userItem.add(LSTM(embeddingdim, embeddingdim, return_sequences=False)) # try using a GRU instead, for fun
	#model.add(Dropout(0.5))
	
	model=Sequential()
	model.add(Merge([model_user_Item_globalItem, model_temp_userItem], mode='concat'))
	model.add(model.add(Dense(2*embeddingdim, embeddingdim))) #embedding *4 x embedding
	# try using different optimizers and different optimizer configs
	model.compile(loss='cos_loss', optimizer='adam') # migth need to implement cosine loss (1-T.dot( y_prd ,y_true) / T.dot(y_prd,y_prd)

	print("Train...")
	model.fit(x_train, y_train, batch_size=1, nb_epoch=4, validation_data=(x_train, y_train), show_accuracy=True)
	#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
	
