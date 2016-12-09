


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
#import random
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.recurrent import GRU,LSTM
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


batchsize=3681
staticFea=300
tempFea=300




def readbatch():
	global staticFile,tempFile,lblFile,batchsize
	static= np.zeros([batchsize,staticFea])
	temp= np.zeros([batchsize,7,tempFea])

	i=0
	while True:
		s=staticFile.readline()
		t=tempFile.readline()
		
		s=unicode(s, errors='ignore')
		t=unicode(t, errors='ignore')
		
		if not s:
			break
		st= s.split('\t')
		j=0
		for si in st:
			if not si or si=='\n':
				j=j-1
			else:
				static[i,	j]=float(si)
			j=j+1
		
		tf=t.split('\t')
		j=0
		tff=np.zeros(7*300)
		for ti in tf:
			if not ti or ti=='\n':
				j=j-1
			else:
				tff[	j]=float(ti)
			j=j+1
		tff=tff.reshape([7,300])
		temp[i]=tff
		i=i+1
		if i==batchsize:
			break
	if i==batchsize:
		hasmore=1
	else:
		hasmore=0
	print(hasmore)	
	return static[0:i-1], temp[0:i-1],hasmore
		
		
staticFile=open(r"\\ZW5338456\F$\newTempOut1\fea_no_user_test.static")
tempFile=open(r"\\ZW5338456\F$\newTempOut1\fea_no_user_test.Temp")

outfile=open(r"\\ZW5338456\F$\newTempOut1\fea_no_user_test.out",'w')

#staticinput ,tempinput,hasmore = readbatch()
staticmodel=Sequential()
staticmodel.add(Dense(300,300))
staticmodel.add(Activation('tanh'))
tempmodel=Sequential()
tempmodel.add(LSTM(tempFea,300))
model=Sequential()
model.add(Merge([staticmodel, tempmodel],mode='concat'))
model.add(Dense(300+300,300))
model.add(Activation('tanh'))





print('done model construction')
model.compile(loss='mean_squared_error', optimizer='Adadelta')
print('done complie')
model.load_weights(r'\\ZW5338456\f$\temprepdiction_no_user_.model.lstm')

j=0
while True:
	print("batch",j)
	j=j+1
	staticinput ,tempinput,hasmore = readbatch()
	ys = model.predict([staticinput ,tempinput])# ,y_train,accuracy=True)# nb_epoch=10, batch_size=1024, verbose=2, show_accuracy=True)
	for y in ys:
		for yi in y:
			outfile.write("%s\t" %yi)
		outfile.write("\n")
	outfile.flush()
	if  hasmore==0 :
		staticFile.close()
		tempFile.close()
		
		break

	
outfile.close()

