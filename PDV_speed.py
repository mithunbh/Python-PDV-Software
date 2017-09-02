'''Written 7/15/2017
Last updated: 8/08/2017

Author: Will Shaw, Xuan Zhou
Translated to python by Larry Salvati

'''
#!/usr/bin/python2.7
#modified for compatibility with python 3.x.x


import matplotlib.pyplot as plt
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
else:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

import numpy as np
from scipy import signal
from scipy import optimize
from numpy import fft

<<<<<<< HEAD

def read_PDV_spectrogram(time_res=15,sample_rate=.08,time_offset=47.9,channel_bool=0):


=======
    
def read_PDV_data(time_res=15,sample_rate=.08,time_offset=47.9,channel_bool=0):
    
    
>>>>>>> 86aaffb7a57641755c760d2b7cd2fb68e251b828
    '''a bunch of functions used later in the program'''
    def read_header(filename):
        with open(filename,'r') as f:
            header = f.readlines()[:6]
            #record_length = header[0].split()[2]
            #sample_interval = header[1].split()[2]
            trigger_point = header[2].split()[4]
            return float(trigger_point) #float(record_length),float(sample_interval),float(Horizontal_offset)


    def read_pdv_data(filename):
        with open(filename,'r') as f:
            f.seek(0)
            amplitude = [];time = [];
            for n, line in enumerate(f):
                if n>6:
                    r= line.split('\t')
                    for i in r:
                        if i in '':
                            r.remove(i)
                    a = float(r[2].split()[0]);t = float(r[1]);
                    time.append(t)
                    amplitude.append(a)
            return amplitude, time

    def next_pow_2(n):
        return n.bit_length()-1

    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
        root = tk.Tk()
        root.withdraw()
        root.pdv_filename = askopenfilename(filetypes=(("Text files","*Ch1.txt"),("All files","*.*")),title="Select file")
        name, ch_ext = root.pdv_filename.split('Ch')
        amplitude = []
    '''actual code begins here'''

    if channel_bool ==1:
        ch_num = 4
    else:
        ch_num = 3
    for i in range(ch_num):
        filename = name + 'Ch'+str(i+1)+'.txt'
        a,t= read_pdv_data(filename)
        amplitude.append(a);
        if i==2:
            trigger_point = read_header(filename)
            for j in t:
                j = j+ abs(trigger_point)
            time = t


    s = [];rms = []
    for i in range(ch_num):
        s.append(0)
        for j in range(int(0.05*len(time))):
            s[i] = s[i] + amplitude[i][j]**2
        rms.append(s[i]/int(0.05*len(time)))

    t_i = []
    for i in range(ch_num):
        for j in range(len(time)):
            if np.sqrt(abs(amplitude[i][j]**2)>=2*rms[i]):
                t_i.append(i)
                break
    fitvalues = [];meanvals = [];camp = []
    t_i = max(t_i)
    for i in range(ch_num):

        fitvalues.append(np.polyfit(time[t_i:len(amplitude[i])],amplitude[i][t_i:len(amplitude[i])],1))
        meanvals.append(np.polyval(fitvalues[i],time))

        if i==0:
            n = .35
        if i==1:
            n = 0.37
        if i==2:
            n = 1
        if i==3:
            n = 0.34

        camp.append([(amplitude[i][j]-meanvals[i][j])/n for j in range(len(amplitude[i]))])


    camp.remove(camp[2]);
    max_trigger = max(amplitude[2])
    time_vector = time[0:amplitude[2].index(max_trigger)]
    for i in amplitude[2]:
        if i>= .9*max_trigger:
            index90 = amplitude[2].index(i)
            break

    time90 = time[index90]*1e9

    time_offset = time_offset-time90;

    sample_freq = 1/(time[1]-time[0]);
    time = [i*1e9 + time_offset for i in time];
    r = int(round(sample_freq*time_res*1e-9));
    test = int(round(sample_rate/.04));
    if test==0:
        test=1
    STFT = []
    for i in range(ch_num-1):
        f,t,Zxx = signal.stft(camp[i][index90-1000:-1],fs = sample_freq,window = signal.hamming(int(r)),nfft = 10*int(r),noverlap = r-test,nperseg = len(signal.hamming(r)));
        STFT.append(Zxx)
    STFT = np.asarray(STFT)

    if channel_bool==1:
        STFT = (np.abs(STFT[0])+np.abs(STFT[1])+np.abs(STFT[2]))/3
    else:
        STFT = (np.abs(STFT[0])+np.abs(STFT[1]))/2


    time_ax = [t[i]*1e9 + time_offset-time[index90-1000] for i in range(len(t))]
    velocity_ax = [i*.775*1e-9 for i in f]
    for t in time_ax:
        if t >=600 and t<601:
            ubound = time_ax.index(t)
    for v in velocity_ax:
        if v>.05 and v<=.06:
            vfilter = velocity_ax.index(v)
        if v>=4:
            vubound = velocity_ax.index(v)
            break
    STFT[0:vfilter,:] = 0





    mx = np.argmax(STFT,axis = 0)
    velocity_lineout = [velocity_ax[mx[i]] for i in range(len(mx))]
    velocity_lineout_fit = velocity_lineout

    for i in range(len(velocity_lineout)):
        if velocity_lineout[i]>0.1 and (mx[i]+2)<len(velocity_ax):
            p = np.polyfit(velocity_ax[mx[i]-2:mx[i]+2],STFT[mx[i]-2:mx[i]+2,i],2)
            velocity_lineout_fit[i] = -float(p[1])/(2.0*float(p[0]))
        else:
            velocity_lineout_fit[i] = 0
<<<<<<< HEAD

    return time, camp, velocity_lineout_fit,time_ax

def find_pdv_speed(time,camp,velocity_lineout_fit,time_ax):
=======
    
    return camp, velocity_lineout_fit,time_ax,

def find_pdv_speed(camp,velocity_lineout_fit):
<<<<<<< HEAD
>>>>>>> 86aaffb7a57641755c760d2b7cd2fb68e251b828
=======
>>>>>>> 86aaffb7a57641755c760d2b7cd2fb68e251b828
    plt.ion()
    h = plt.figure(2)
    plt.plot(time_ax[0:ubound],velocity_lineout_fit[0:ubound],marker = 'o',linestyle = ':',markersize = 1)
    plt.title("select beginning and end of launch")
    pi,pf = plt.ginput(2)
    plt.title("Velocity lineout")
    h.show()





    minv = min(np.abs([velocity_lineout_fit[j]-pf[1] for j in range(len(velocity_lineout_fit))]))
    mint = min(np.abs([time_ax[j]-(pi[0]) for j in range(len(time_ax))]))
    maxt = min(np.abs([time_ax[j]-(pf[0]) for j in range(len(time_ax))]))
    for i in range(len(time_ax)):
        if np.abs(time_ax[i]-(pi[0]))==mint:
            ti = time_ax[i]
        if np.abs(time_ax[i]-(pf[0]))==maxt:
            tf = time_ax[i]

    ti_min = min(np.abs([time[i]-ti for i in range(len(time))]))
    tf_min = min(np.abs([time[i]-tf for i in range(len(time))]))
    for i in range(len(time)):
        if np.abs(time[i]-ti)==ti_min:
            ti = time[i]
        if np.abs(time[i]-tf)==tf_min:
            tf = time[i]

    for i in range(len(velocity_lineout_fit)):
        if np.abs(velocity_lineout_fit[i]-pf[1])==minv:
            speed = velocity_lineout_fit[i]
            break



    ti_ind = time.index(ti)
    tf_ind = time.index(tf)
    camp = np.asarray(camp)



    L = tf_ind-ti_ind
    NFFT = 2**next_pow_2(int(L)*10)

    Y1 = fft.fft(camp[0,ti_ind:tf_ind],NFFT)
    Y2 = fft.fft(camp[1,ti_ind:tf_ind],NFFT)
    if channel_bool ==1:
        Y3 = fft.fft(camp[2,ti_ind:tf_ind],NFFT)
        Y = (Y1+Y2+Y3)/(3*L)
    else:
        Y = (Y1+Y2)/(2*L)
    f = (((time[1]-time[0])*1e-9)**-1)/2*np.linspace(0,1,NFFT/2+1)*.775*1e-9

    Yshort = 2*np.abs(Y[0:int(NFFT/2)+1])
    yfilt = []
    ffilt = []
    for i in range(len(f)):
        if f[i]> 0.5*speed and f[i]<1.5*speed:
            yfilt.append(Yshort[i])
            ffilt.append(f[i])

    yfilt = np.asarray(yfilt)
    ffilt = np.asarray(ffilt)


    plt.figure(3)
    plt.plot(f,2*np.abs(Y[0:int(NFFT/2)+1]))
    plt.show()

    mean = sum(ffilt*yfilt)/len(ffilt)
    sigma = sum(yfilt*(ffilt - mean)**2)/len(ffilt)
    try:
        popt,pcov = optimize.curve_fit(gaus,ffilt,yfilt)
        plt.plot(ffilt,gaus(ffilt,*popt))
        print("peak velocity =" + str(popt[1])+"km/s")
        perr = np.sqrt(np.diag(pcov))
        perr = perr[1]*np.sqrt(len(ffilt))*np.sqrt(2)
        print("estimated gaussian fit error = " + str(perr))
    except RuntimeError:
        print("unable to optimize gaussian fit")
        print("Max Lineout velocity= " + str(speed))
<<<<<<< HEAD
<<<<<<< HEAD

=======
=======
>>>>>>> 86aaffb7a57641755c760d2b7cd2fb68e251b828
    
>>>>>>> 86aaffb7a57641755c760d2b7cd2fb68e251b828
    plt.figure(1)
    plt.imshow(STFT[0:vubound,0:ubound],aspect = "auto",origin="lower",extent = [time_ax[0],time_ax[ubound],0,velocity_ax[vubound]],cmap = "seismic",interpolation = "bicubic")

    plt.figure(5)
    plt.plot(np.asarray(time_2)*1e9,amplitude[2])
    plt.show()
    plt.figure(4)
    plt.waitforbuttonpress()
    return Y,popt,perr,ffilt,gaus(ffilt,*popt)










'''the main execution loop. Put user defined variables in here'''
if __name__ == "__main__":
    '''takes variables (with their defaults):
    time_res=15 value in nanoseconds
    sample_rate=.08
    time_offset=47.9 (time offset, determined empirically after alignment)
    channel_bool=0   (boolean to determine if we have 3 working channels or not. 1 means 3 channels, and actually anything else means 2.
                        keep in mind that this program runs under the assumption that scope channel 4 is the broken channel.
                        if this ever changes, it might require a little logic manipulation.)
    '''
    time, camp, velocity_lineout_fit,time_ax=read_PDV_spectrogram(channel_bool=1)
    find_pdv_speed()
