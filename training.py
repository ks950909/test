# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os

# 3 read csv

def read_csv(output = 4):
    #root folder name
    root_folder_name = './train/'

    #each folder name
    train_folder_name = root_folder_name+'Train/'
    valid_folder_name = root_folder_name+'valid/'
    test_folder_name = root_folder_name+'test/'

    trainx, trainy = read_csv_mode(train_folder_name)
    print('train read fin')
    validx, validy = read_csv_mode(valid_folder_name)
    print('valid read fin')
    testx, testy = read_csv_mode(test_folder_name)
    print('test read fin')

    

    
    if output == 4:
        return trainx,trainy,validx,validy
    else:
        return trainx,trainy,validx,validy,testx,testy
def read_csv_mode(folder_name):
    
    # each x,y file
    #x_file_name = folder_name + 'x.csv'
    #y_file_name = folder_name + 'y.csv'
    #x_file_name = folder_name + 'x_nn.csv'
    #y_file_name = folder_name + 'y_nn.csv'
    x_file_name = folder_name + 'x_big.csv'
    y_file_name = folder_name + 'y_big.csv'

    
    tempx = pd.read_csv(x_file_name)
    outputx = np.array(tempx)[1:,1:]
    outputx = outputx.reshape(-1,129,126,1)

    tempy = pd.read_csv(y_file_name)
    outputy = np.array(tempy)[:,1:]

    
    return outputx,outputy

# 4 make_model

# if no upside np,  add np
#import numpy as np
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

# make_model : output-model
# In here no data input
def make_model( learning_rate = 1e-3):

    # plz setting diff
    nb_filters = 64
    nb_pool = 2
    nb_conv = 5
    #
    
    #make model
    model = Sequential()
    #Layer1 shape : (? ,rows ,cols , 1)
    model.add(BatchNormalization())
    model.add(Conv2D(nb_filters,kernel_size=(nb_conv,nb_conv) , activation = 'relu' , input_shape = (129,126,1) ))
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    
    
    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv2D(nb_filters, kernel_size=(nb_conv, nb_conv) , activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    
    #Layer3 
    model.add(BatchNormalization())
    model.add(Conv2D(nb_filters, kernel_size=(nb_conv, nb_conv) , activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))
    
    #Layer4
    model.add(BatchNormalization())
    model.add(Conv2D(nb_filters, kernel_size=(nb_conv, nb_conv) , activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool) ))
    model.add(Dropout(0.25))

    
    #final 
    model.add(Flatten())
    #model.add(BatchNormalization())
    #model.add(Dense(60,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    adam = Adam(lr=learning_rate)
    # loss change
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    

    return model
    
#5 hyperparameter tuning 

from sklearn.metrics import log_loss
import keras

# plt
# if not have delete
import matplotlib.pyplot as plt
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# to here


#%matplotlib inline


#from keras import backend as K

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
    def on_epoch_end(self,batch,logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def hyperparameter_tuning():
    
    batch_size = 15
    nb_epoch = 43
    lr_op = 0.144669

    # list_size : each k how many loop
    list_size = 10
    
    X_train , Y_train , X_valid , Y_valid = read_csv(4)
    
    # make model (if not pyplot do comment
    #model = make_model(0.02)
    #plot_model(model, show_shapes= True,to_file='model.png')
    #
    
    
    # acc_maxrenew each lr_max,epoch_max,batch_max renew
    acc_max = 0.0
    lr_max = 0.0
    epoch_max = 0
    batch_max = 0
    
    # tuning 
    # 1. random pick 
    # 2. near by good case
    # mix 1,2

    # learning rate is float , others int
    # every thing bigger than 0
    for k in range(20):
        #if k >= 10:
        #    lr_list_temp = 0.001 * 10 ** np.random.uniform(0, 2.5, list_size)
        #    epoch_list_temp = np.random.randint(1, 200, list_size)
        #    batch_list_temp = np.random.randint(1, 200, list_size)
        #else:
        #    lr_list_temp = lr_op * 10 ** np.random.uniform(-2.0/((k+1)**2), 2.0/((k+1)**2), list_size)
        #    epoch_list_temp = nb_epoch + np.random.randint(-k*4, k*5+1, list_size)
        #    batch_list_temp = batch_size + np.random.randint(-k*1-1, k*5, list_size)
        lr_list_temp = 0.001 * 10 ** np.random.uniform(0, 1, list_size)
        epoch_list_temp = np.random.randint(1, 500, list_size)
        batch_list_temp = np.random.randint(1, 1000, list_size)
        for i in range(list_size):
            # model make by lr,ep,ba
            model = make_model(lr_list_temp[i])
            # I don't take hist to plot ... plz obtain plot method

            hist = model.fit(X_train,Y_train,batch_size = batch_list_temp[i],epochs= epoch_list_temp[i]
                      , shuffle = True, verbose = 0, validation_data = (X_valid,Y_valid))
            
            # I don't know why loss and accuracy diff
            loss,accuracy = model.evaluate(X_valid,Y_valid,verbose = 1)

            #prediction_valid = model.predict(X_valid,batch_size = batch_size, verbose =1)
            #score = log_loss(Y_valid,prediction_valid)

            # show plt
            #plt.figure()
            #plt.xlabel('Epochs')
            #plt.ylabel('acc')
            #plt.plot(hist['acc'])
            #plt.plot(hist['val_acc'])
            #plt.legend(['Training', 'Validation'])
            # end plt

            # print logging
            print('for : ',k,' lr ',lr_list_temp[i],' ep ',epoch_list_temp[i],' ba ',batch_list_temp[i],' ac ',accuracy*100)
            # renew
            if acc_max < accuracy:
                lr_max = lr_list_temp[i]
                epoch_max = epoch_list_temp[i]
                batch_max = batch_list_temp[i]
                acc_max = accuracy

            # print now max
            print('lrmax:',lr_max ,' epochmax:', epoch_max ,' batchmax:', batch_max , ' accmax: ',acc_max*100 )    
            
    # print max val
    print('lrmax:',lr_max ,' epochmax:', epoch_max ,' batchmax:', batch_max , ' accmax: ',acc_max)    
    return lr_max


# 6 train

# read csv is diff to hyperparameter 
# 6 input read csv
# not yet fin
def train():
   
    # from 5
    batch_size = 33
    nb_epoch = 46
    learning_rate = 0.005637

    X_train , Y_train , X_valid , Y_valid ,X_test, Y_test = read_csv(6)
    
    model = make_model(learning_rate)
    # make model (if not pyplot do comment)
    plot_model(model, show_shapes= True,to_file='models.png')
    #
    
        
    
    model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),)
    
    loss,accuracy = model.evaluate(X_test,Y_test,verbose = 1)
    prediction_valid = model.predict(X_test,batch_size = batch_size, verbose =1)
    score = log_loss(Y_test,prediction_valid)
    
    print(accuracy)

def main():
    #hyperparameter_tuning()
    train()
    


if __name__=='__main__':
    main()

