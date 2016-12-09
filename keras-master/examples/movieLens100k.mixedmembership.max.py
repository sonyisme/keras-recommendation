


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano

np.random.seed(1337) # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,TimeDistributedDense,MaxDot,Merge,Reshape,ElementMul,MaxTopic,Cosine
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




#print(len(user), 'train sequences',r"f:\1b.items.n0",)


userModel1 = Sequential()
userModel1.add(Dense(1682, 300))
userModel1.add(Activation('tanh'))
userModel1.add(Dropout(0.4))
userModel1.add(Dense(300, 100))
userModel1.add(Activation('tanh'))

itemModel1 = Sequential()
itemModel1.add(TimeDistributedDense(943, 300))
itemModel1.add(Activation('tanh'))
itemModel1.add(Dropout(0.4))
itemModel1.add(TimeDistributedDense(300, 100))
itemModel1.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model1=Sequential()
model1.add(MaxDot([userModel1,itemModel1])) #should output 2 values 
model1.add(Reshape(2,1))

userModel2 = Sequential()
userModel2.add(Dense(1682, 300))
userModel2.add(Activation('tanh'))
userModel2.add(Dropout(0.4))
userModel2.add(Dense(300, 100))
userModel2.add(Activation('tanh'))

itemModel2 = Sequential()
itemModel2.add(TimeDistributedDense(943, 300))
itemModel2.add(Activation('tanh'))
itemModel2.add(Dropout(0.4))
itemModel2.add(TimeDistributedDense(300, 100))
itemModel2.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model2=Sequential()
model2.add(MaxDot([userModel2,itemModel2])) #should output 2 values 
model2.add(Reshape(2,1))


userModel3 = Sequential()
userModel3.add(Dense(1682, 300))
userModel3.add(Activation('tanh'))
userModel3.add(Dropout(0.4))
userModel3.add(Dense(300, 100))
userModel3.add(Activation('tanh'))

itemModel3 = Sequential()
itemModel3.add(TimeDistributedDense(943, 300))
itemModel3.add(Activation('tanh'))
itemModel3.add(Dropout(0.4))
itemModel3.add(TimeDistributedDense(300, 100))
itemModel3.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model3=Sequential()
model3.add(MaxDot([userModel3,itemModel3])) #should output 2 values 
model3.add(Reshape(2,1))

userModel4 = Sequential()
userModel4.add(Dense(1682, 300))
userModel4.add(Activation('tanh'))
userModel4.add(Dropout(0.4))
userModel4.add(Dense(300, 100))
userModel4.add(Activation('tanh'))

itemModel4 = Sequential()
itemModel4.add(TimeDistributedDense(943, 300))
itemModel4.add(Activation('tanh'))
itemModel4.add(Dropout(0.4))
itemModel4.add(TimeDistributedDense(300, 100))
itemModel4.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model4=Sequential()
model4.add(MaxDot([userModel4,itemModel4])) #should output 2 values 
model4.add(Reshape(2,1))

userModel5 = Sequential()
userModel5.add(Dense(1682, 300))
userModel5.add(Activation('tanh'))
userModel5.add(Dropout(0.4))
userModel5.add(Dense(300, 100))
userModel5.add(Activation('tanh'))

itemModel5 = Sequential()
itemModel5.add(TimeDistributedDense(943, 300))
itemModel5.add(Activation('tanh'))
itemModel5.add(Dropout(0.4))
itemModel5.add(TimeDistributedDense(300, 100))
itemModel5.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model5=Sequential()
model5.add(MaxDot([userModel5,itemModel5])) #should output 2 values 
model5.add(Reshape(2,1))



userModel6 = Sequential()
userModel6.add(Dense(1682, 300))
userModel6.add(Activation('tanh'))
userModel6.add(Dropout(0.4))
userModel6.add(Dense(300, 100))
userModel6.add(Activation('tanh'))

itemModel6 = Sequential()
itemModel6.add(TimeDistributedDense(943, 300))
itemModel6.add(Activation('tanh'))
itemModel6.add(Dropout(0.4))
itemModel6.add(TimeDistributedDense(300, 100))
itemModel6.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model6=Sequential()
model6.add(MaxDot([userModel6,itemModel6])) #should output 2 values 
model6.add(Reshape(2,1))


userModel7 = Sequential()
userModel7.add(Dense(1682, 300))
userModel7.add(Activation('tanh'))
userModel7.add(Dropout(0.4))
userModel7.add(Dense(300, 100))
userModel7.add(Activation('tanh'))

itemModel7 = Sequential()
itemModel7.add(TimeDistributedDense(943, 300))
itemModel7.add(Activation('tanh'))
itemModel7.add(Dropout(0.4))
itemModel7.add(TimeDistributedDense(300, 100))
itemModel7.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model7=Sequential()
model7.add(MaxDot([userModel7,itemModel7])) #should output 2 values 
model7.add(Reshape(2,1))

userModel8 = Sequential()
userModel8.add(Dense(1682, 300))
userModel8.add(Activation('tanh'))
userModel8.add(Dropout(0.4))
userModel8.add(Dense(300, 100))
userModel8.add(Activation('tanh'))

itemModel8 = Sequential()
itemModel8.add(TimeDistributedDense(943, 300))
itemModel8.add(Activation('tanh'))
itemModel8.add(Dropout(0.4))
itemModel8.add(TimeDistributedDense(300, 100))
itemModel8.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model8=Sequential()
model8.add(MaxDot([userModel8,itemModel8])) #should output 2 values 
model8.add(Reshape(2,1))

userModel9 = Sequential()
userModel9.add(Dense(1682, 300))
userModel9.add(Activation('tanh'))
userModel9.add(Dropout(0.4))
userModel9.add(Dense(300, 100))
userModel9.add(Activation('tanh'))

itemModel9 = Sequential()
itemModel9.add(TimeDistributedDense(943, 300))
itemModel9.add(Activation('tanh'))
itemModel9.add(Dropout(0.4))
itemModel9.add(TimeDistributedDense(300, 100))
itemModel9.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model9=Sequential()
model9.add(MaxDot([userModel9,itemModel9])) #should output 2 values 
model9.add(Reshape(2,1))

userModel10 = Sequential()
userModel10.add(Dense(1682, 300))
userModel10.add(Activation('tanh'))
userModel10.add(Dropout(0.4))
userModel10.add(Dense(300, 100))
userModel10.add(Activation('tanh'))

itemModel10 = Sequential()
itemModel10.add(TimeDistributedDense(943, 300))
itemModel10.add(Activation('tanh'))
itemModel10.add(Dropout(0.4))
itemModel10.add(TimeDistributedDense(300, 100))
itemModel10.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model10=Sequential()
model10.add(MaxDot([userModel10,itemModel10])) #should output 2 values 
model10.add(Reshape(2,1))

userModel11 = Sequential()
userModel11.add(Dense(1682, 300))
userModel11.add(Activation('tanh'))
userModel11.add(Dropout(0.4))
userModel11.add(Dense(300, 100))
userModel11.add(Activation('tanh'))

itemModel11 = Sequential()
itemModel11.add(TimeDistributedDense(943, 300))
itemModel11.add(Activation('tanh'))
itemModel11.add(Dropout(0.4))
itemModel11.add(TimeDistributedDense(300, 100))
itemModel11.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model11=Sequential()
model11.add(MaxDot([userModel11,itemModel11])) #should output 2 values 
model11.add(Reshape(2,1))

userModel12 = Sequential()
userModel12.add(Dense(1682, 300))
userModel12.add(Activation('tanh'))
userModel12.add(Dropout(0.4))
userModel12.add(Dense(300, 100))
userModel12.add(Activation('tanh'))

itemModel12 = Sequential()
itemModel12.add(TimeDistributedDense(943, 300))
itemModel12.add(Activation('tanh'))
itemModel12.add(Dropout(0.4))
itemModel12.add(TimeDistributedDense(300, 100))
itemModel12.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model12=Sequential()
model12.add(MaxDot([userModel12,itemModel12])) #should output 2 values 
model12.add(Reshape(2,1))

userModel13 = Sequential()
userModel13.add(Dense(1682, 300))
userModel13.add(Activation('tanh'))
userModel13.add(Dropout(0.4))
userModel13.add(Dense(300, 100))
userModel13.add(Activation('tanh'))

itemModel13 = Sequential()
itemModel13.add(TimeDistributedDense(943, 300))
itemModel13.add(Activation('tanh'))
itemModel13.add(Dropout(0.4))
itemModel13.add(TimeDistributedDense(300, 100))
itemModel13.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model13=Sequential()
model13.add(MaxDot([userModel13,itemModel13])) #should output 2 values 
model13.add(Reshape(2,1))

userModel14 = Sequential()
userModel14.add(Dense(1682, 300))
userModel14.add(Activation('tanh'))
userModel14.add(Dropout(0.4))
userModel14.add(Dense(300, 100))
userModel14.add(Activation('tanh'))

itemModel14 = Sequential()
itemModel14.add(TimeDistributedDense(943, 300))
itemModel14.add(Activation('tanh'))
itemModel14.add(Dropout(0.4))
itemModel14.add(TimeDistributedDense(300, 100))
itemModel14.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model14=Sequential()
model14.add(MaxDot([userModel14,itemModel14])) #should output 2 values 
model14.add(Reshape(2,1))

userModel15 = Sequential()
userModel15.add(Dense(1682, 300))
userModel15.add(Activation('tanh'))
userModel15.add(Dropout(0.4))
userModel15.add(Dense(300, 100))
userModel15.add(Activation('tanh'))

itemModel15 = Sequential()
itemModel15.add(TimeDistributedDense(943, 300))
itemModel15.add(Activation('tanh'))
itemModel15.add(Dropout(0.4))
itemModel15.add(TimeDistributedDense(300, 100))
itemModel15.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model15=Sequential()
model15.add(MaxDot([userModel15,itemModel15])) #should output 2 values 
model15.add(Reshape(2,1))




userModel16 = Sequential()
userModel16.add(Dense(1682, 300))
userModel16.add(Activation('tanh'))
userModel16.add(Dropout(0.4))
userModel16.add(Dense(300, 100))
userModel16.add(Activation('tanh'))

itemModel16 = Sequential()
itemModel16.add(TimeDistributedDense(943, 300))
itemModel16.add(Activation('tanh'))
itemModel16.add(Dropout(0.4))
itemModel16.add(TimeDistributedDense(300, 100))
itemModel16.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model16=Sequential()
model16.add(MaxDot([userModel16,itemModel16])) #should output 2 values 
model16.add(Reshape(2,1))


userModel17 = Sequential()
userModel17.add(Dense(1682, 300))
userModel17.add(Activation('tanh'))
userModel17.add(Dropout(0.4))
userModel17.add(Dense(300, 100))
userModel17.add(Activation('tanh'))

itemModel17 = Sequential()
itemModel17.add(TimeDistributedDense(943, 300))
itemModel17.add(Activation('tanh'))
itemModel17.add(Dropout(0.4))
itemModel17.add(TimeDistributedDense(300, 100))
itemModel17.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model17=Sequential()
model17.add(MaxDot([userModel17,itemModel17])) #should output 2 values 
model17.add(Reshape(2,1))

userModel18 = Sequential()
userModel18.add(Dense(1682, 300))
userModel18.add(Activation('tanh'))
userModel18.add(Dropout(0.4))
userModel18.add(Dense(300, 100))
userModel18.add(Activation('tanh'))

itemModel18 = Sequential()
itemModel18.add(TimeDistributedDense(943, 300))
itemModel18.add(Activation('tanh'))
itemModel18.add(Dropout(0.4))
itemModel18.add(TimeDistributedDense(300, 100))
itemModel18.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model18=Sequential()
model18.add(MaxDot([userModel18,itemModel18])) #should output 2 values 
model18.add(Reshape(2,1))

userModel19 = Sequential()
userModel19.add(Dense(1682, 300))
userModel19.add(Activation('tanh'))
userModel19.add(Dropout(0.4))
userModel19.add(Dense(300, 100))
userModel19.add(Activation('tanh'))

itemModel19 = Sequential()
itemModel19.add(TimeDistributedDense(943, 300))
itemModel19.add(Activation('tanh'))
itemModel19.add(Dropout(0.4))
itemModel19.add(TimeDistributedDense(300, 100))
itemModel19.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model19=Sequential()
model19.add(MaxDot([userModel19,itemModel19])) #should output 2 values 
model19.add(Reshape(2,1))

userModel20 = Sequential()
userModel20.add(Dense(1682, 300))
userModel20.add(Activation('tanh'))
userModel20.add(Dropout(0.4))
userModel20.add(Dense(300, 100))
userModel20.add(Activation('tanh'))

itemModel20 = Sequential()
itemModel20.add(TimeDistributedDense(943, 300))
itemModel20.add(Activation('tanh'))
itemModel20.add(Dropout(0.4))
itemModel20.add(TimeDistributedDense(300, 100))
itemModel20.add(Activation('tanh'))
##itemModel.add(Reshape(4))
##itemModel.add(Dense(4, 2))
model20=Sequential()
model20.add(MaxDot([userModel20,itemModel20])) #should output 2 values 
model20.add(Reshape(2,1))




model=Sequential()
model.add(Merge([model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11,model12,model13,model14,model15,model16,model17,model18,model19,model20],mode='concat'))
#model.add(MaxTopic()),
model.add(TimeDistributedDense(20,1))
#model.add(TimeDistributedDense(300, 1))
##model.add(Activation('normalization'))
model.add(Reshape(2))
y_score= model.get_output(train=False)
x_test=model.get_input(train=False)
model.add(Activation('softmax'))
##model.add(Merge([userModel, itemModel], mode='sum'))


print('done model construction')
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')

print("Loading data...")
user ,Items, y_train = load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.users100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_pos100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.items_neg100k0",45915)
print('done complie')
scoring= theano.function(x_test,y_score,allow_input_downcast=True, mode=None)
history = model.fit([user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items,user ,Items] ,y_train, nb_epoch=20, batch_size=2048, verbose=2, show_accuracy=True)

#history = model.train_on_batch([user ,Items] ,y_train,accuracy=True)# nb_epoch=10, batch_size=1024, verbose=2, show_accuracy=True)
print('done training')
user_test ,Items_test, y_test = load_dataset(r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.userstest100k.centered",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k",r"C:\Users\t-alie\Downloads\movieLens_1M\movielens.itemstest100k.fakeneg",45915)
y_p=model.custom_predict([user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test,user_test ,Items_test],scoring)
#y_pp=model.predict([user_test,Items_test])
pfile=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp_cos_max20","w")
for y in y_p:
	pfile.write("%s\n" %y)
pfile.close()
#pfile1=open(r"C:\Users\t-alie\Downloads\movieLens_1M\yp1","w")
#for y in y_pp:
#	pfile1.write("%s\n" %y)

#pfile1.close()
print('done prediction')
model.save_weights(r'f:\maxtopic20.model')
#print('done saving')