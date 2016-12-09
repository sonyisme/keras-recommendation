


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

def load_dataset():
	fitems = open(r"F:\1b.items.p", 'r')
	fitemsN = open(r"f:\1b.items.n0", 'r')
	fusers = open(r"f:\1b.users", 'r')
	
	lusers=fusers.readline()
	user= np.zeros([1024,3883])
	items= np.zeros([1024,2,6039])
	y_train=np.zeros([1024,2])

	i=0
	for lusers in fusers:
		
		litems=fitems.readline()
		litemsN=fitemsN.readline()
		
		feats=lusers.split(" ")
		pfeats=litems.split(" ")
		nfeats=litemsN.split(" ")

		for fea in feats:
			if ':' in fea:
				x=fea.split(":")
				id=int(unicode(x[0], errors='ignore'))-1
				user[i][id]=float(unicode(x[1], errors='ignore') )
				y_train[i][0]=1

		
		for fea in pfeats:
			if ':' in fea:
				x=fea.split(":")
				id=int(unicode(x[0], errors='ignore'))-1
				items[i][0][id]=float(unicode(x[1], errors='ignore') )

		for fea in nfeats:
			if ':' in fea: 
				x=fea.split(":")
				id=int(unicode(x[0], errors='ignore'))-1
				items[i][1][id]=float(unicode(x[1], errors='ignore') )

		i=i+1
	
	#user=   np.array([[1,0],[1,0],[0,1]])
	#y_train=np.array([[1,0],[1,0],[1,0]])	
	#Items=np.array(  [ [[1,0],[0,1]] , [[.5,0],[0,1]],[[-1,1],[1,0]] ])
	#user=   np.array([[1,1,1],[1,3,1],[0,1,0],[0,2,-1]])
	#y_train=np.array([[1,0],[1,0],[1,0],[1,0]])	
	#Items=np.array(  [ [[1,2,0],[0,2,0]] , [[2,2,1],[2,0,2]],[[0,1,2],[1,0,0]],[[1,3,3],[1,3,-1]] ])
	#user=   np.array([[0,1]])
	#y_train=np.array([[1,0]])	
	#Items=np.array(  [[[-1,1],[1,0]]])
	# The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
	#user.reshape(-1,3);
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
	return (user ,items, y_train)


print("Loading data...")
user ,Items, y_train = load_dataset()
print(len(user), 'train sequences')

print('user_train shape:', user.shape)
print('Item shape:', Items.shape)
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
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model=Sequential()
model.add(Cosine([userModel,itemModel])) #should output 2 values 
##model.add(Activation('normalization'))
model.add(Reshape(2))
##model.add(Merge([userModel, itemModel], mode='sum'))


print('done model construction')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
print('done complie')
history = model.fit([user ,Items] ,y_train, nb_epoch=10, batch_size=1, verbose=2, show_accuracy=True)
print('done training')
#model.save_weights(r'f:\1b.model')
#print('done saving')