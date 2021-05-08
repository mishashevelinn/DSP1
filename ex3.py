import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

plots = 5

#  calculating fourier series on a single period
def Fourier_syn(sig, a0, an, m, N, xL, xR):
    tt = np.linspace(0, 2, N // 5)
    FT = np.zeros((len(tt)), dtype=complex)
    FT += eval(a0)
    for k in range(-m, m + 1):
        if k == 0:
            continue
        else:
            cn = eval(an)
            FT += cn * np.exp(1j * np.pi * k * tt)
    return FT


xL = -2
xR = 8
N = 1000
a0 = '(np.exp(1) - 1) / 2'
an = '((np.exp(1)*((-1)**k)) - 1) / (2*(1 - 1j*np.pi*k))'
tt = np.linspace(xL, xR, N)
sig = np.zeros((len(tt)))

# generating periodic function
for i in range(0, len(sig), N // 5):
    sig[i:i + np.round(int(N / 10))] += np.exp(tt[i:i + np.round(int(N / 10))] % 1)

plt.subplot(plots, 1, 1)
plt.plot(tt, sig)

# generating periods for Fourier series
FT = Fourier_syn(sig, a0, an, 200, N, xR, xL)
FT_periodic = np.zeros((tt.shape), dtype=complex)

for i in range(0, len(FT_periodic), len(FT)):
    FT_periodic[i: i + len(FT)] += FT

# plotting periodic Fourier series
plt.subplot(plots, 1, 2)
plt.plot(tt, FT_periodic)

# beth

# amplitude is a radius of a complex Fourier coefficient
freq = np.arange(-7, 8)
amp = [np.abs(eval(an)) for k in range(-7, 8)]
plt.subplot(plots, 1, 3)
plt.bar(freq, amp, color='maroon',width=0.1)

# phase shift is an angle of complex Fourier coefficient
plt.subplot(plots, 1, 4)
plt.bar(freq, [np.angle(eval(an)) for k in range(-7, 8)], width=0.1)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')



#   fourier series for x(-t) + 1 using the same
def Fourier_syn2(sig, a0, an, m, N, xL, xR):
    tt = np.linspace(3, 1, N // 5)
    FT = np.zeros((len(tt)), dtype=complex)
    FT += eval(a0) + 1
    for k in range(-m, m + 1):
        if k == 0:
            continue
        else:
            cn = eval(an)
            FT += cn * np.exp(1j * np.pi * k * tt)
    return FT


Yt = Fourier_syn2(sig, a0, an, 100, N, xR, xL)

FTy_periodic = np.zeros((tt.shape), dtype=complex)
for i in range(0, len(FTy_periodic), len(Yt)):
    FTy_periodic[i: i + len(Yt)] += Yt
plt.subplot(plots, 1, 5)
plt.plot(tt, FTy_periodic)
plt.show()


def lin_chirp(f1, f2, dur, fs = 44100):

    fs = 10000
    dur = 10
    f1 = 100
    mu = (f2 - f1) / (2*dur)  # calculating Mu
    A = 2
    phi = -np.pi / 3
    tt = np.linspace(0, dur, dur * fs + 1)
    xx = A * np.cos(2 * np.pi * mu * tt ** 2 + +2 * np.pi * f1 * tt + phi)
    S = np.abs(librosa.stft(xx))
    plt.figure()
    librosa.display.specshow(S, x_axis='time', y_axis='linear', sr=fs)
    plt.colorbar()
    plt.show()
    sd.play(xx, fs)
lin_chirp()


def sampler(A, f0, phi, fs):
    tt = np.arange(-10, 10, 1 / fs)  # given analytical data - analog sig,
    # sampling it by given frequency

    epsilon = 1 / len(tt)  # finding zero on x axis
    zero = tt[(tt < epsilon) & (tt > 0)]
    zero_index = np.argwhere(tt == zero[0])[0][0]

    sampled_sig = np.zeros(len(tt))  # allocating array for sampled signal

    for k in range(len(tt)):  # simulating sampling process
        sampled_sig[k] = A * np.cos(2 * np.pi * f0 * tt[k] + phi)

    xL = zero_index - 2 * (fs // f0)  # calculating indexes of sliced range:
    xR = zero_index + 5 * (fs // f0)  # two periods before zero index and five after

    # plotting block
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('Sampled signal properties', fontsize=16)
    fig.tight_layout(pad=2)
    ax[0].set_xlabel('time', fontsize=20)
    ax[0].set_ylabel('real amplitude', fontsize=20)
    # plotting block end

    ax[0].stem(tt[xL:xR], sampled_sig[xL:xR])  # plotting sliced sampled signal

    sampled_freq = (2 * np.pi * f0) / fs  # normalized frequency of sampled signal
    freq = np.arange(-6 * np.pi, 7 * np.pi, sampled_freq / 2)  # building frequency axis
    amps = np.zeros(len(freq))
    phaseShifts = np.zeros(len(freq))

    #  finding all complex frequencies including aliases - 2π shifts of original frequencies
    index = [np.argwhere(freq == sampled_freq + np.pi * i)[0][0] for i in range(-6, 8, 2)]
    index2 = [np.argwhere(freq == -sampled_freq + np.pi * i)[0][0] for i in range(-4, 8, 2)]

    # building complex amplitudes and phase shifts array
    for k, j in zip(index, index2):
        amps[k] = A / 2
        amps[j] = A / 2
        phaseShifts[k] = phi
        phaseShifts[j] = -phi

    # plotting block
    labels = ['0' if i == 0 else str(i) + 'π' for i in range(-6, 7)]
    ax[1].set_xlabel('frequency', fontsize=20)
    ax[1].set_ylabel('complex amplitude', fontsize=20)
    ax[1].set_xticks([freq[i] for i in range(0, len(freq) - 1, 5)])
    ax[1].set_xticklabels(labels)
    # plotting block end

    ax[1].stem(freq, amps)

    # plotting block
    ax[2].set_xlabel('frequency', fontsize=20)
    ax[2].set_ylabel('phase shift', fontsize=20)
    ax[2].set_xticks([freq[i] for i in range(0, len(freq) - 1, 5)])
    ax[2].set_yticks((-np.pi / 6, -np.pi / 3, 0, np.pi / 6, np.pi / 3))
    ylabels = ['-π/6', '-π/3', '0', 'π/6', 'π/3']
    ax[2].set_yticklabels(ylabels)
    ax[2].set_xticklabels(labels)
    # plotting block end

    ax[2].stem(freq, phaseShifts)

    plt.show()


sampler(7, 100, -np.pi / 3, 500)
