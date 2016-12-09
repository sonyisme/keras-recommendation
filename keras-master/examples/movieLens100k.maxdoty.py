


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano

np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul,MaxDot
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def inspect_inputs(i, node, fn):
	print( "Beging intput:")
	print (i)
	print (node)
	print ("input(s) value(s):")
	print ([input[0].shape for input in fn.inputs])
	theano.printing.debugprint(node)
	print( "End input:")

def inspect_outputs(i, node, fn):
	print( "Beging output:")
	print( "output(s) :", [output[0].shape for output in fn.outputs])
	print( "End output:")

def load_dataset(userFile,posFile,negFile,n):
	fitems = open(posFile, 'r')
	fitemsN = open(negFile, 'r')
	fusers = open(userFile, 'r')
	
	lusers=fusers.readline()
	user= np.zeros([n,1682])
	items= np.zeros([n,2,943])
	y_train=np.zeros([n,2])

	i=0
	for lusers in fusers:
		if i==n:
			break
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
user ,Items, y_train = load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.users100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_pos100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_neg100k0",45915)

#print(len(user), 'train sequences',r"f:\1b.items.n0",)

print('user_train shape:', user.shape)
print('Item shape:', Items.shape)
userModel = Sequential()
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
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model=Sequential()
model.add(MaxDot([userModel,itemModel])) #should output 2 values 
#model.add(TimeDistributedDense(300, 1))
##model.add(Activation('normalization'))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))
##model.add(Merge([userModel, itemModel], mode='sum'))


print('done model construction')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
print('done complie')
scoring= theano.function(x_test,y_score,allow_input_downcast=True, mode=None)
history = model.fit([user ,Items] ,y_train, nb_epoch=100, batch_size=2048, verbose=2, show_accuracy=True)

#history = model.train_on_batch([user ,Items] ,y_train,accuracy=True)# nb_epoch=10, batch_size=1024, verbose=2, show_accuracy=True)
print('done training')
user_test ,Items_test, y_test = load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.userstest100k.centered",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k.fakeneg",50781)
y_p=model.custom_predict([user_test,Items_test],scoring)
#y_pp=model.predict([user_test,Items_test])
pfile=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp_max","w")
for y in y_p:
	pfile.write("%s\n" %y)
pfile.close()
#pfile1=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp1","w")
#for y in y_pp:
#	pfile1.write("%s\n" %y)

#pfile1.close()
print('done prediction')
#model.save_weights(r'f:\1b.model')
#print('done saving')