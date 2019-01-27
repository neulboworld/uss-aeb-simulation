import numpy as np
from scipy import signal
from scipy.signal import freqs, butter, lfilter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs

    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def generatePulse(Phase, Tc, Ts, nPulse):
    t = np.arange(int(Tc/Ts * nPulse)) * Ts
    s = np.sin(2*np.pi/Tc*t + np.pi/180*Phase)
    Pulse = (s > 0) * 2 - 1

    return Pulse


def DQPSKmodulation(code, codeSize, Tc, Ts, nPulse):

    if codeSize % 2 == 1:
        ModCode = code << 1

    SymbolLen = Tc * nPulse
    SymbolSize = int(np.ceil(codeSize/2) + 1)

    t = np.arange(int(Tc/Ts * nPulse)) * Ts
    ModulatedPulse = np.sin(2*np.pi/Tc*t)
    Phase = 0

    for index in np.flip(range(SymbolSize - 1)):
        if ModCode & (2**(index + SymbolSize - 1)):
            if ModCode & (2**index):
                Phase = Phase + 45
            else:
                Phase = Phase + 315

        else:
            if (codeSize % 2 == 1) and (index >= SymbolSize - 2):
                Phase = Phase + 180
            elif ModCode & (2**index):
                Phase = Phase + 135
            else:
                Phase = Phase + 225

        Phase = Phase % 360

        Delay = SymbolLen * index
        Pulse = np.sin(2*np.pi/Tc*(t-Delay) + np.pi/180*Phase)
        ModulatedPulse = np.append(ModulatedPulse, Pulse)

    ModulatedPulse = (ModulatedPulse > 0) * 2 - 1

    return ModulatedPulse


def generateCodeStream(code, SamplePerBit):
    InphaseBitSequence = np.array([])
    QuadBitSequence = np.array([])

    if code.size % 2 == 1:  # add NULL bit if ODD LENGTH
        code = np.append(code, 0)

    code = np.reshape(code, (2, int(code.size / 2)))
    SymbolLen = int(SamplePerBit * code.size / 2)

    for SampleNum in range(SymbolLen):
        Index = int(SampleNum/SamplePerBit)
        InphaseBitSequence = np.insert(InphaseBitSequence, SampleNum, code[0, Index])
        QuadBitSequence = np.insert(QuadBitSequence, SampleNum, code[1, Index])

    BitSequence = np.array([InphaseBitSequence, QuadBitSequence])
    # bit_seq_I = np.append(BitSequence, code[int(sample / n_sample_bit)])
    # bit_seq_Q = np.append(bit_seq_Q, code[int(np.ceil(code.size/2) + sample / n_sample_bit)])
    return BitSequence


if __name__ == "__main__":

    # Parameter and code set-up
    fc = 48 * pow(10, 3)
    fs = 1.25 * pow(10, 6)
    Ts = 1/fs
    Tc = 1/fc

    fcut =  3*pow(10,3)
    BW = 4000

    # Generate TX / RX signal 
    # Amplitude = 12
    # EbN0dB = np.arange(-10,51,1)
    # SNR = pow(10,EbN0dB/10)
    # N0 = np.sqrt(2) * pow(Amplitude,2) / SNR
    # AttenuationCoef = -1.46017

    # CodeHex = 0x766B8C7F
    # CodeSize = 31

    CodeHex = 0x72
    CodeSize = 7

    nPulsePerBit = 12
    nPulseSample = fs / fc
    nSymbolSample = nPulseSample * nPulsePerBit
    nCodeSample = nSymbolSample * CodeSize

    # Transmission
    TxPulse = DQPSKmodulation(CodeHex, CodeSize, Tc, Ts, nPulsePerBit)
    TxPulse = butter_bandpass_filter(TxPulse, fc - BW/2, fc + BW/2, fs, 1)

    # Receiving
    AddDelay = np.append(np.zeros(1000), TxPulse)
    AddDelay = np.append(AddDelay, np.zeros(1000))
    RcvSig = AddDelay
    SymbolDelay = int(nSymbolSample)

    # Demodulation
    MulI = RcvSig[SymbolDelay:] * RcvSig[:-SymbolDelay]
    MulQ = signal.hilbert(RcvSig[SymbolDelay:]) * RcvSig[:-SymbolDelay]

    DemodI = butter_lowpass_filter(MulI, fcut, fs, 4)
    DemodQ = butter_lowpass_filter(MulQ, fcut, fs, 4)

    # Plotting
    plt.subplot(211)
    plt.plot(RcvSig[SymbolDelay:])
    plt.subplot(212)
    plt.plot(signal.hilbert(RcvSig[SymbolDelay:]))
    plt.show()
