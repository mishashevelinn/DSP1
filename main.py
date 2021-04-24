# import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def cosgen(fs, A, f0, dur, phi=0):
    Ts = 1 / fs
    tt = np.arange(0, dur, Ts)
    xt = A * np.cos(2 * np.pi * f0 * tt + phi)
    return tt, xt


def ampPlot():
    t = np.linspace(-180, 180, 1000)
    t1 = lambda t: (np.sqrt((900 ** 2) + (t ** 2))) / (3 * (10 ** 8))
    t2 = lambda t: (55 + (np.sqrt(((t - 55) ** 2) + (900 ** 2)))) / (3 * (10 ** 8))
    freq = 2*np.pi*180*(10**6)

    amp = np.abs(np.exp(-1j*(freq)*t1(t)) + np.exp(-1j*(freq)*(t2(t))))

    plt.subplot(2,1,1)
    plt.plot(t, amp)
    plt.subplot(2,1,2)
    plt.show()
    # plt.magnitude_spectrum(summ)
    # plt.show()
ampPlot()


tt, xt = cosgen(44100, 5, 460, 4)

#peaks, _ = find_peaks(xt+xt1, prominence=1)


for i in range(420, 454, 2):
    tt, xt1 = cosgen(44100, 5, i, 4)
    #plt.plot(peaks, (xt+xt1)[peaks], 'ob')
    summ = xt+xt1
    plt.plot(summ[:4000])
    plt.show()
    #print(peaks)
    peak = np.amax(summ)
    result = np.where(summ == 0)
    print(result)
    print(peak)
    print((result[0][3] - result[0][2]))


# for i in range(len(peaks)):
#     for j in range(i + 1, len(peaks)):
#         if summ[peaks[i]] == peak and summ[peaks[j]] == peak:
#             periods.append( ((peaks[j] - peaks[i])/44100))
#             print((4*(peaks[j] - peaks[i])/44100))
#             break