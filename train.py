# -*- coding: utf-8 -*-



from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
%matplotlib inline

from keras import backend as K


def make_model(rows , cols , learning_rate = 1e-3):

    
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    


    
    #make model
    model = Sequential()
    #Layer1 shape : (? ,129 ,126 , 1)
    #model.add(ZeroPadding2D( (1,1) ))
    model.add(Conv2D(nb_filters,(nb_conv,nb_conv),padding = 'same' , activation = 'relu' , input_shape = (129,126,1) ))
    #want use leaky relu
    
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    '''
    #Layer2 shape : (? , , , )
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv) , activation = 'relu'))
    #batch nomaliztion
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    
    #Layer3 shape : (? , , , )
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv) , activation = 'relu'))
    #batch nomaliztion
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    #Final shape : (? , , , )
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv) , activation = 'relu'))
    #batch nomaliztion
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    model.add(Flatten())
    '''
    
    #Add Fully Connected Layer
    
    #model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1000, activation='softmax'))
    
    #load imagenet pre=trained data
    #model.load_weights('')
    
    #truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(1000, activation='softmax'))
    
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
    
def train():
   
    rows , cols = 224,224
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10
    learning_rate = 1e-3
    #X_train , Y_train , X_valid , Y_valid = data.load_data(rows,cols)
    
    
        
    model = make_model(rows,cols,learning_rate)
    
    
    
    SVG(model_to_dot(model,show_shapes = True).create(prog='dot',format='svg'))
    return
    model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),)
    
    prediction_valid = model.predict(X_valid,bach_size = batch_size, verbose =1)
    score = log_loss(Y_valid,prediction_valid)
    sigdata = data.wav2np()
    
    specgram = Spectrogram(n_dft=512, n_hop=128, input_shape=(len_src, 1))
    
    #model.add(BatchNormalization(axis=time_axis)) # recommended
    #model.add(your_awesome_network)
    
def hyperparameter_tuning():
    rows , cols = 224,224
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10
    
    X_train , Y_train , X_valid , Y_valid = data.load_data(rows,cols)
    
    lr_op = 0.02
    fc_op = 900
    
    for i in range(10):
        lr_list_temp = lr_op * 10 ** np.random.uniform(-2.0/((k+1)**2), 2.0/((k+1)**2), 10)
        fc_list_temp = fc_op + np.random.randint(-(10-k)*50, (10-k)*50, 10)
        
    model = make_model(rows,cols,learning_rate)
    
    
def test():
    model = Sequential()

    model.add(Conv2D(2, (3, 3), padding='same', activation='relu', input_shape=(8, 8, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    #plot_model(model, to_file='model.png')
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))





def main():
    #mnist_test()
    #train()
    test()
if __name__ == '__main__':
    main()
