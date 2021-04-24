# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9   14:49:34 2021

@author: signal processing class
"""

import matplotlib.pyplot as plt 
import numpy as np
import sounddevice as sd
import time 
#from scipy.signal import hamming

def cosgen(fs, A, f0, phi, dur, plot_flag = 1):
    """
    cosgen generates and plot a sinusoidal signal with a given amplitude, 
    frequency, phase and duration
    Input arguments fs - sampling frequency, A - amplitude, f0 - frequency
    phi - phase shift, dur - duration, plotflag - 1 - plot, 0
    cosgen returns xt and tt - the signal and the corresponding time axis
    

    """
    Ts = 1/fs
    tt = np.arange(0, dur, Ts)

    xt = A*np.cos(2*np.pi*f0*tt+phi)
    if plot_flag == 1:
        plt.plot(tt,xt, 'b-', tt,xt, 'g.') 
        plt.xlabel('x'), plt.ylabel('Amplitude')
        plt.title('A*cos(2*pi*f0*t+ phi)')
        plt.grid()
        plt.show()
    
    return xt, tt

def Energy(sig, fs, frame_length, hop_size):
    """
    
    """    
    siglen = sig.size
    print(siglen)
    N = int(np.ceil(siglen/hop_size))
    E=np.zeros(N)    
    j=0
    y = np.zeros(int(siglen))    
    for i in range(0, siglen, hop_size):
        if i>=siglen:
            break
        E[j] = 1/frame_length*np.sum(sig[i:i+frame_length] ** 2)
        y[i:i+frame_length] = E[j]
        j = j + 1
    E = smooth(E, window_len=5 )
    y = smooth(y, window_len = hop_size*2)
    return y, E

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
 


# beat notes experiment
# tuning fork signal
fs = 10000
A = 1
f1 = 440
phi = 0 #-np.pi*0.4
dur = 2
(xx1, tt1) = cosgen(fs, A, f1, phi, dur)
#sd.play(xx1, fs)
time.sleep(dur)
T5 = np.int(np.round(5/f1*fs))
plt.plot(tt1[:T5], xx1[:T5])


# "piano" signal
fs = 10000
A = 1
f2 = 435
phi = 0 #-np.pi*0.4
dur = 2
(xx2, tt2) = cosgen(fs, A, f2, phi, dur)
#sd.play(xx2, fs)
time.sleep(dur)
T5 = np.int(np.round(5/f2*fs))
plt.plot(tt2[:T5], xx2[:T5])

# sum of tuning fork and "piano"
xxsum = xx1 + xx2
f_delta = abs((f2 - f1)/2)
ndelta5 = int(np.round(5/(2*f_delta)*fs)) # duration of 5 beats in samples
plt.plot(tt1[:ndelta5], xxsum[:ndelta5])
#sd.play(xxsum, fs)

# computing the (smoothed) short-time energy of xxsum
y, E = Energy(xxsum, fs, 256, 128)
plt.figure()
plt.plot(E[:ndelta5])
plt.figure
plt.plot(E)









    
    
    
    
