#
# Libraries
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack 
import cmath
from scipy import signal
from keras.utils import to_categorical
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
plt.style.use('classic')

#-----------------------------------------------------------------------------#

# Set variables
# Signal
sampling_rate = 10000
freq = 1000              # Not used if using shift = 'frequency'
bps = 100                # bits per sec
total_bits = 10000       # Total number of bits desired
total_time = total_bits/bps

# Spectrogram variables
frame_length = 50
frame_overlap = 25
window = 'hanning'
fft_size = 1024            

# Independent loop variables
shift = ['amplitude','frequency','phase']         
desired_SNR = [-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21]

num_tests = 1   # Total number of tests
#-----------------------------------------------------------------------------#

# Funtions for spectrogram generation 
#
def enframe(x,S,L,wdw):
    # Divides the time series signal 'x' into multiple frames of length 'L' 
    # with frame overlap (L-S) and applies window 'wdw'
    # Outputs the windowed frames
    w = signal.get_window(wdw,L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0,nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return(frames)
def stft(frames,N,Fs):
    # Does short term fourier transform of length 'N' to each frame of 'frames' 
    # containing time-series data of sampling frequency 'Fs'
    # Outputs the complex, magnitude, real, imaginary and phase spectra of each frame
    stft_frames = np.array([ fftpack.fft(x,N) for x in frames])
    #freq_axis = np.linspace(0,Fs,N)
    mag_spectra = np.array([ abs(x) for x in stft_frames ])
    real_spectra = np.array([ x.real for x in stft_frames ])
    imag_spectra = np.array([ x.imag for x in stft_frames ])
    phase_spectra = []
    for i in range(0,len(stft_frames)):
        phase_spectra.append([ cmath.phase(x) for x in stft_frames[i] ])
    phase_spectra = np.array(phase_spectra)
    return(stft_frames, mag_spectra, real_spectra, imag_spectra, phase_spectra)

#-----------------------------------------------------------------------------#

                              # Main
# Choosing modulation type
for a in range(num_tests):
    for k in shift:
        # Setting 'high' and 'low' for each modulation type
        if k=='amplitude':
            h=1
            l=0.7
        elif k=='frequency':
            h=1000
            l=950
        elif k=='phase':
            h=0
            l=25
        else:
            print('Please set shift=amplitude,phase or frequency')
            exit()
        dig_sig = [l,h]
        x_comb = []
        # Creating a modulated signal
        for i in range(len(dig_sig)):
            # Creating time domain signal
            t = np.linspace(0,bps/sampling_rate,int(sampling_rate/bps))
            if k=='phase':
                x = np.sin(2*np.pi*freq*t+(dig_sig[i]*np.pi/180))
            elif k=='amplitude':
                x = dig_sig[i]*np.sin(2*np.pi*freq*t)
            else:
                x = np.sin(2*np.pi*dig_sig[i]*t)
            x_comb = np.concatenate((x_comb,x))
        #-------------------------------------------------------------------------#
        # Creating copies of x_comb to get to total_bits 
        x_comb = np.reshape(x_comb,(len(dig_sig),len(x)))
        sig_td = np.array([x_comb]*int(total_bits/len(dig_sig)))
        sig_td = np.reshape(sig_td,(total_bits,len(x)))
        label = np.array([dig_sig]*int(total_bits/len(dig_sig)))
        label = label.flatten()
        
        # Shuffling sig_td and label in the same order
        temp = list(zip(sig_td, label)) 
        random.shuffle(temp) 
        sig_td, label = zip(*temp)
        sig_td = (np.array(sig_td))
        sig_td_flat = sig_td.flatten()
        label = np.array(label)
        # Injecting AWGN 
        for j in desired_SNR:
            variance_sig_td = np.var(sig_td_flat)
            variance_noise = variance_sig_td/10**(j/10)
            mean_noise = 0
            noise =  np.random.normal(mean_noise, np.sqrt(variance_noise), len(sig_td_flat))
            noise_sig_td = sig_td_flat + noise 
            #-------------------------------------------------------------------------#
            # Creating different types of spectrograms
            mag_spectra, real_imag_spectra = ([] for i in range(2))
            for l in range(len(label)):
                start = int(l*sampling_rate/bps)
                stop = int((l+1)*sampling_rate/bps)
                win_frames = enframe(noise_sig_td[start:stop], frame_length-frame_overlap, frame_length, window)
                stft_frames_x, mag_spectra_x, real_spectra_x, imag_spectra_x, phase_spectra_x = stft(win_frames, fft_size, sampling_rate)
                mag_spectra.append(mag_spectra_x)
                real_imag_x = np.dstack((real_spectra_x,imag_spectra_x))
                real_imag_spectra.append(real_imag_x)
            mag_spectra = np.array(mag_spectra)
            real_imag_spectra = np.array(real_imag_spectra)
            #-----------------------------------------------------------------------------#
            # Scaling data
            noise_sig_td_scaled = preprocessing.scale(noise_sig_td)    
            timedomain_noisy_signal_new = np.reshape(noise_sig_td_scaled,(total_bits,1,int(sampling_rate/bps)))
            mag_spectra_scaled = preprocessing.scale(mag_spectra.flatten())
            mag_spectra_new = np.reshape(mag_spectra_scaled,mag_spectra.shape)
            real_imag_spectra_scaled = preprocessing.scale(real_imag_spectra.flatten())
            real_imag_spectra_new = np.reshape(real_imag_spectra_scaled,real_imag_spectra.shape)
            #-----------------------------------------------------------------------------#
            choice_NN = [timedomain_noisy_signal_new,mag_spectra_new,real_imag_spectra_new]
            # Neural Network
            for m in range(len(choice_NN)):
                start_time = time.time()
                
                dataset = choice_NN[m]   
                # Setting and encoding label (necessary for to_categorial below)
                le = preprocessing.LabelEncoder()
                label_encoded = le.fit_transform(np.ravel(label))
                
                # Splitting dataset into training and testing sets
                train_images, test_images, train_labels, test_labels = train_test_split(dataset, label_encoded, test_size=0.3,random_state=1, stratify=label_encoded)
                
                # Turning training and testing labels to binary class matrix
                train_labels_cat = to_categorical(train_labels, num_classes=2)
                test_labels_cat = to_categorical(test_labels, num_classes=2)
                
                # Creating a model
                model = models.Sequential()
                
                # Neural network 
                model.add(layers.Flatten())
                model.add(layers.Dense(64, activation='relu'))
                model.add(layers.Dense(2, activation='softmax'))
            
                # Fitting the model with training data
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(train_images, train_labels_cat, epochs=10, batch_size=16)
                
                # Training time
                stop_time = time.time()
                total_time = stop_time-start_time
                print('Total time taken: %.3f' % total_time)
                print('shift:',k)
                print('SNR:',j)
                print('dataset:',m)
                
                # Performance
                # predict probabilities for test set
                yhat_probs = model.predict(test_images, verbose=0)
                # predict crisp classes for test set
                yhat_classes = model.predict_classes(test_images, verbose=0)
                
                test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
                print('Testing accuracy:', test_acc)
                print('Testing loss:', test_loss)          
                precision = precision_score(test_labels, yhat_classes)
                print('Precision: %f' % precision)
                recall = recall_score(test_labels, yhat_classes)
                print('Recall: %f' % recall)
                f1 = f1_score(test_labels, yhat_classes)
                print('F1 score: %f' % f1)
                           
                
                # Writing the best results to a text file
                file = open("Experiment results_test10", "a+")
                file.write('%s,%.2f,%i,%.5f,%.5f,%.5f,%.5f,%.5f \n' 
                           %(k,j,m,test_acc,test_loss,precision,recall,f1))
                file.close()   
                '''
                # Full results to a text file
                save_info =[np.concatenate(([j],[m],
                                            [test_acc],[test_loss],[total_time],
                                            history.history['accuracy'],history.history['loss']))]
                file = open("Experiment results_test10_full.txt", "a+")
                np.savetxt(file, save_info, delimiter=',')
                file.close() 
                '''        
                del(start_time,stop_time,total_time,dataset,le,label_encoded,train_images, 
                    test_images,train_labels, test_labels,model,test_loss,test_acc,history,
                    precision,recall,f1,yhat_probs,yhat_classes,
                    test_labels_cat,train_labels_cat)
                
            del(noise_sig_td,variance_sig_td,variance_noise,mean_noise,noise,
                    mag_spectra,real_imag_spectra,start,stop,win_frames,stft_frames_x,
                    mag_spectra_x,real_spectra_x,imag_spectra_x, phase_spectra_x,real_imag_x,
                    noise_sig_td_scaled,timedomain_noisy_signal_new,mag_spectra_scaled,
                    mag_spectra_new,real_imag_spectra_scaled,real_imag_spectra_new,
                    choice_NN)
        del(x_comb,dig_sig,h,l,temp,label,sig_td_flat,sig_td,t,x)
        
#----------------------------------------------------------------------------------------#




