


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
#import random
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul
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
curline=0
batchsize=2048
userfea=1682
itemfea=943
samples=0

lusers=[]
litems=[]
litemsN=[]
rows=[]
def load_dataset(userFile,posFile,negFile, randomize=0):
	fitems = open(posFile, 'r')
	fitemsN = open(negFile, 'r')
	fusers = open(userFile, 'r')
	global lusers,litems,litemsN,samples,curline,rows
	lusers=fusers.readlines()
	litems=fitems.readlines()
	litemsN=fitemsN.readlines()
	samples=len(lusers)
	rows=range(0,samples-1)
	if randomize==1:
		np.random.shuffle(rows)
		
		print(rows[0:10])
	curline=0
	
def readbatch():
	global curline
	n=min(batchsize, samples-curline )
	user= np.zeros([n,userfea])
	items= np.zeros([n,2,itemfea])
	y_train=np.zeros([n,2])

	i=0
	for row in rows[curline:curline+n]:
		if i==n:
			break
		
		luser=lusers[row]
		litem=litems[row]
		litemN=litemsN[row]
		
		feats=luser.split(" ")
		pfeats=litem.split(" ")
		nfeats=litemN.split(" ")

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
	curline=curline+n
	return (user ,items, y_train)


print("Loading data...")
load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.users100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_pos100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_neg100k0",1)
print(samples)
#print(len(user), 'train sequences',r"f:\1b.items.n0",)

#print('user_train shape:', user.shape)
#print('Item shape:', Items.shape)
userModel = Sequential()
userModel.add(Dense(1682, 300))
userModel.add(Activation('tanh'))
userModel.add(Dropout(0.4))
userModel.add(Dense(300, 300))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(943, 300))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.4))
itemModel.add(TimeDistributedDense(300, 300))
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
##model.add(Merge([userModel, itemModel], mode='sum'))


print('done model construction')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
print('done complie')
scoring= theano.function(x_test,y_score,allow_input_downcast=True, mode=None)
#history = model.fit([user ,Items] ,y_train, nb_epoch=15, batch_size=2048, verbose=2, show_accuracy=True)
for i in range(0,3):
	print("itr",i)
	for j in range(0,int(samples/batchsize+.05)):
		print("batch",j)
		user ,Items, y_train = readbatch()
		history = model.train_on_batch([user ,Items] ,y_train,accuracy=True)# nb_epoch=10, batch_size=1024, verbose=2, show_accuracy=True)
	curline=0;

print('done training')
load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.userstest100k.centered",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k.fakeneg")
pfile=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp_cos.batch","w")
for j in range(0,int(samples/batchsize+.05)):
	print("testing batch",j)
	user ,Items, y_train = readbatch()
	y_p=model.custom_predict([user,Items],scoring)
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