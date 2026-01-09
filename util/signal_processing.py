from scipy import signal


def bandpass_filter(data, lowcut, highcut, fs, order=4, filter_type='butter'):
    """
    Apply bandpass filter to data

    Parameters:
    data: numpy array (n_samples, n_channels)
    lowcut: low cutoff frequency (Hz)
    highcut: high cutoff frequency (Hz)
    fs: sampling frequency (Hz)
    order: filter order
    filter_type: 'butter' (Butterworth) or 'fir' (FIR filter)

    Returns:
    Filtered data
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    if filter_type == 'butter':
        # Butterworth filter
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=0)
    elif filter_type == 'fir':
        # FIR filter
        numtaps = order * 2 + 1
        taps = signal.firwin(numtaps, [lowcut, highcut],
                             pass_zero=False, fs=fs)
        filtered_data = signal.filtfilt(taps, 1.0, data, axis=0)

    return filtered_data
