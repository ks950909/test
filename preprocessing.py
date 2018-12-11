import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from keras.utils import normalize

#each folder name
root_folder_name = './train/'
train_folder_name = root_folder_name+'Train/'
valid_folder_name = root_folder_name+'valid/'
test_folder_name = root_folder_name+'test/'

# array of value
value_array = ['down','go','left','no','off','on','right','stop','up','yes']


def get_file_list(root_folder_name):
    output_file_list = []
    root_folder=os.listdir(root_folder_name)
    

    for folder_name in root_folder:
        full_folder_name = root_folder_name  +folder_name 
        if(not os.path.isdir(full_folder_name)):
            continue
        file_list = os.listdir(full_folder_name)
        for file_name in file_list:
            full_file_name = full_folder_name + '/' + file_name
            output_file_list.append([full_file_name,folder_name,file_name])
    return output_file_list
def file_save(folder_name):
    print(folder_name)
    y_value=[]
    allaboutthis=np.arange(129*126).reshape((1,129*126))
    filelist = get_file_list(folder_name)
    file_len = len(filelist)
    file_idx = 0
    for filename in filelist:
        tempdata = wav2np(filename[0])
        tempdata = tempdata.reshape((1,129*126))
        allaboutthis=np.concatenate((allaboutthis,tempdata))
        for i in range(10):
            if filename[1] == value_array[i] :
                a=[0]*10
                a[i]=1
                y_value.append(a)
                break
            if i == 9:
                print('error - no found folder name')
        file_idx = file_idx+1
        if(file_idx % 100 == 0):
            print(file_idx,'/',file_len)
        
    dataframe=pd.DataFrame(allaboutthis)
    # filename change x
    #dataframe.to_csv(folder_name+'x_nn.csv')# no nomalize
    dataframe.to_csv(folder_name+'x.csv')# nomalize
    #
    dataframe2=pd.DataFrame(y_value)
    # finename change y
    #dataframe2.to_csv(folder_name+'y_nn.csv')# no nomalize
    #
    dataframe2.to_csv(folder_name+'y.csv')# nomalize

def wav2np(filename):
    sample_rate, samples = wavfile.read(filename)
    frequencies, times, spectrogram = signal.stft(samples, fs=sample_rate)
    spectrogram = np.abs(spectrogram)
    # nomalize
    #spectrogram = normalize(spectrogram)
    if len(frequencies) != 129:
        print("fre : ",filename," ",len(frequencies))
    if len(times) > 126:
        print("tim : ",filename," ",len(times))
    
    if len(spectrogram[0]) is not 126:
        a=np.full((129,126-len(spectrogram[0])),0.0)
        spectrogram = np.hstack((spectrogram,a))
    return spectrogram

def main():
    file_save(train_folder_name)
    print('train fin')
    file_save(valid_folder_name)
    print('valid fin')
    file_save(test_folder_name)
    print('all fin')

if __name__=='__main__':
    main()

