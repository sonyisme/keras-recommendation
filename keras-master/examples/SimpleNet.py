from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,Cosine,Merge,Reshape,ElementMul
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer



import theano

def inspect_inputs(i, node, fn):
	print (i)
	print (node)
	print ("input(s) value(s):")
	print ([input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
	print( "output(s) value(s):", [output[0] for output in fn.outputs])


def load_dataset():
	#user=   np.array([[1,0],[1,0],[0,1]])
	#y_train=np.array([[1,0],[1,0],[1,0]])	
	#Items=np.array(  [ [[1,0],[0,1]] , [[.5,0],[0,1]],[[-1,1],[1,0]] ])
	user=   np.array([[1,1,1],[1,3,1],[0,1,0],[0,2,-1]])
	y_train=np.array([[1,0],[1,0],[1,0],[1,0]])	
	Items=np.array(  [ [[1,2,0],[0,2,0]] , [[2,2,1],[2,0,2]],[[0,1,2],[1,0,0]],[[1,3,3],[1,3,-1]] ])
	#user=   np.array([[0,1]])
	#y_train=np.array([[1,0]])	
	#Items=np.array(  [[[-1,1],[1,0]]])
	# The inputs come as vectors, we reshape them to monochrome 2D images,
    # according to the shape convention: (examples, channels, rows, columns)
	user.reshape(-1,3);
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
	return (user ,Items, y_train)


print("Loading data...")
user ,Items, y_train = load_dataset()
print(len(user), 'train sequences')

print('user_train shape:', user.shape)
print('Item shape:', Items.shape)
userModel = Sequential()
userModel.add(Dense(3, 3))
userModel.add(Activation('tanh'))
userModel.add(Dropout(0.1))
userModel.add(Dense(3, 2))
userModel.add(Activation('tanh'))

itemModel = Sequential()
itemModel.add(TimeDistributedDense(3, 3))
itemModel.add(Activation('tanh'))
itemModel.add(Dropout(0.1))
itemModel.add(TimeDistributedDense(3, 2))
itemModel.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model=Sequential()
model.add(ElementMul([userModel,itemModel])) #should output 2 values 
model.add(TimeDistributedDense(2, 1))
##model.add(Activation('normalization'))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))
##model.add(Merge([userModel, itemModel], mode='sum'))


print('done model construction')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
print('done complie')
scoring= theano.function(x_test,y_score,
            allow_input_downcast=True, mode=None)
history = model.fit([user ,Items] ,y_train, nb_epoch=5, batch_size=1024, verbose=2, show_accuracy=True)

#history = model.train_on_batch([user ,Items] ,y_train,accuracy=True)# nb_epoch=10, batch_size=1024, verbose=2, show_accuracy=True)
print('done training')
#user_test ,Items_test, y_test = load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.userstest100k.centered",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k.fakeneg",50781)
y_p=model.predict([user,Items])
y_pp=model.custom_predict([user,Items],scoring)
print('done score compile')

pfile=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp","w")
for y in y_p:
	pfile.write("%s\n" %y )

for y in y_pp:
	pfile.write("%s\n" %y )


pfile.close()

print('done prediction')
#model.save_weights(r'f:\1b.model')