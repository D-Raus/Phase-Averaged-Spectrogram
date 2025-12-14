"""
Functions to compute the phase-averaged spectrogram of a signal

D-Raus (21/11/05)
"""
import numpy as np
from scipy import signal
import scipy as sp


import pyTSA



def PhaseAveraged_spectrogram_MeanCycleLength(data,ind_min_approximate,window,overlap,nfft,fs,sens):
    """
    Function phase-averaging a cyclic noise
    This function can be used if all the cycles have close duration
   
    data: microphone signal
    ind_min_approximate: indexes of cycles beginning
    window,overlap,nfft,fs = fft parameters
    sens: far-field microphone sensitivity 
    
    """
    ### Initialization of spectrograms shape
    _,_,PSD_CL_init = signal.spectrogram(data[ind_min_approximate[1]:ind_min_approximate[2]], fs,mode='psd',nperseg= window,noverlap =  overlap,window = 'hann',nfft=nfft,axis=- 1);
    f_CL = np.zeros(PSD_CL_init.shape[0])
    t_CL = np.zeros(PSD_CL_init.shape[1])
    PSD_CL = np.zeros(PSD_CL_init.shape)
    
    mm=0
    
    for ii in range(2,len(ind_min_approximate)-1):
    
        ### Compute the spectrogram of the signal on each cycle
        f_CL_tmp, t_CL_tmp, PSD_CL_tmp = signal.spectrogram(data[ind_min_approximate[ii]:ind_min_approximate[ii+1]], fs,mode='psd',nperseg=window,noverlap = overlap,window = 'hann',nfft=nfft,axis=- 1)       
        f_CL = f_CL_tmp + f_CL
        t_CL = t_CL_tmp + t_CL
        PSD_CL = PSD_CL + PSD_CL_tmp/(sens**2)
    
        mm = mm+1
        
    
    ### Average all the spectrograms
    t_CL = np.array(t_CL)
    f_CL = np.array(f_CL)
    PSD_CL = np.array(PSD_CL)
    t_CL_mean = t_CL/mm                   # Phase-averaged time vector
    f_CL_mean = f_CL/mm                   # Phase-averaged freqency vector
    PSD_CL_mean = PSD_CL/mm               # Phase-averaged Power Spectral density array

    return PSD_CL_mean, t_CL_mean, f_CL_mean


def PhaseAveraged_spectrogram_TSA(data,t,ind_min,ind_fmax,window,overlap,nfft,fs,sens,method):
    """
    Function phase-averaging a cyclic noise
    This function can be used if all the cycles have different durations
    
    data: microphone signal
    ind_min: indexes of cycles beginning
    window,overlap,nfft,fs = fft parameters
    sens: far-field microphone sensitivity
    method: "Time domain" or "Frequency domain"
    """           
    ### Compute the spectrogram of the entire signal.
    f_PSD,t_PSD,PSD = signal.spectrogram(data, fs,mode='psd',nperseg= window,noverlap =  overlap,window = 'hann',nfft=nfft,axis=- 1);

    ### Calculate the indexes of the cycles beginning with the spectrogram time steps
    ind_min_spectrogram = np.zeros(ind_min.shape,dtype=int)
    for ii in np.arange(len(ind_min)):
        ind_min_spectrogram[ii] = np.abs(t_PSD-t[ind_min[ii]]).argmin()
    
    if method == "Time domain":
        
        py_TSA_res = [pyTSA.pyTSA_TimeDomain(PSD[ff,:],t_PSD,ind_min_spectrogram,fs) for ff in np.arange(ind_fmax)]
        
        ### extract the Phase-averaged PSD and time
        y_TSA_TimeDomain = np.squeeze(np.array([py_TSA_res[ff][0] for ff in np.arange(ind_fmax)]))
        t_interp = np.squeeze(np.array([py_TSA_res[ff][1] for ff  in np.arange(ind_fmax)]))
        
        ### calibrate
        y_TSA = y_TSA_TimeDomain/(sens**2)
    
    elif method == "Frequency domain":
        
        
        py_TSA_res = [pyTSA.pyTSA_fft(PSD[ff,:],t_PSD,ind_min_spectrogram,fs) for ff in np.arange(ind_fmax)]
        
        ### extract the Phase-averaged PSD and time
        y_TSA_fft = np.squeeze(np.array([py_TSA_res[ff][0] for ff in np.arange(ind_fmax)]))
        t_interp = np.squeeze(np.array([py_TSA_res[ff][1] for ff  in np.arange(ind_fmax)]))
        
        ### calibrate
        y_TSA = y_TSA_fft/(sens**2)
    
    t_interp = t_interp[0,:]-t_interp[0,0]
    
    return y_TSA,t_interp,f_PSD


def OASPL_fromSpectrogram(PSD_mean,f_mean,t_mean,f_min_OASPL,f_max_OASPL):
    """
    Function to compute the Overall Sound Pressure Level of a phase-averaged spectrogram
    
    t_mean: Phase-averaged time vector
    f_mean: Phase-averaged frequency vector
    PSD_mean: Phase-averaged PSD array
    f_min_OASPL, f_min_OASPL: frequency limits for the integral
    OASP: Overal Sound Pressure Level (dB)
    """
    _, ind_f_min = min((val, idx) for (idx, val) in enumerate(abs(f_mean - f_min_OASPL)))     # Indice correspondant   f_min_OASPL
    _, ind_f_max = min((val, idx) for (idx, val) in enumerate(abs(f_mean - f_max_OASPL)))     # Indice correspondant   f_max_OASPL
    
    OASPL=np.zeros(len(t_mean))
    for jj in range(len(t_mean)):
        OASPL[jj] = np.trapz(PSD_mean[ind_f_min:ind_f_max,jj],f_mean[ind_f_min:ind_f_max])
     
    return OASPL              



def detection_cycles(AOA,cycle1):
    """
    Function to detect the index of the beginning of the oscillation cycles
    AOA: Vector containing the angle of attack
    cycle1: first cycle to detect
    """
    ind_min,_ = sp.signal.find_peaks(-(AOA-15), height=14,distance=10000)    # Detect minimums of angle of attack
    
    # If the cycles have close durations, one can choose to use a constant cycle length
    # corresponding to the mean cycle length of all the oscillations
    nb = np.arange(0,len(ind_min)-cycle1);
    ind_min_approximate = ind_min[cycle1] + int(np.mean(np.round(np.diff(ind_min[cycle1:]))))*nb;
    ind_min_approximate = [int(x) for x in ind_min_approximate ]                  # indice du debut des cycles (debut du mouvement ascendant)
   
    return ind_min,ind_min_approximate
