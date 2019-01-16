import numpy as np
from scipy import signal
from scipy.signal import freqs, butter, lfilter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs

    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
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

def hex2bin(hexVal,size):
	array = np.array([])
	for i in range(size):
		array = np.append(array, ((hexVal & pow(2,i)) >> i) * 2 -1)

	array = np.flip(array)
	return array

def generatePulse(Phase, Tc, Ts, nPulse):
	t = np.arange(int(Tc/Ts * nPulse)) * Ts
	s = np.sin(2*np.pi/Tc*t)

	Pulse = np.array([])
	
	for i in s:
		if i >= 0:
			Pulse = np.append(Pulse, 1)
		else:
			Pulse = np.append(Pulse, -1)

	nDelaySample = int((Phase / 360) * (Tc/Ts))
	
	if nDelaySample > 0:
		Pulse = np.append(Pulse[nDelaySample:], Pulse[:nDelaySample])

	return Pulse

def DQPSKmodulation(code, Tc, TS, nPulse):

	ModulatedPulse = np.array([])

	if code.size % 2 == 1: # add NULL bit if ODD LENGTH
		code = np.append(code, 0)

	code = np.reshape(code, (2, int(code.size / 2)) ) 
	phaseArray = np.array([0])
	phase = 0

	for index in range(np.size(code,1)):
		if code[(0,index)] == 1:
			if code[(1,index)] == 1:
				phase = phase + 45
			elif code[(1,index)] == -1:
				phase = phase + 315

		elif code[(0,index)] == -1:
			if code[(1,index)] == 1:
				phase = phase + 135
			elif code[(1,index)] == -1:
				phase = phase + 225
			else:
				phase = phase + 180

		phase = phase % 360

		#phase = phase + (135 - code[(0,index)] * 90) * code[(1,index)]
		phaseArray = np.append(phaseArray, phase)

	for p in phaseArray:
		temp = generatePulse(p, Tc, Ts, nPulse)
		ModulatedPulse = np.append(ModulatedPulse, temp)

	return ModulatedPulse


def generateCodeStream(code, SamplePerBit):
	InphaseBitSequence = np.array([])
	QuadBitSequence = np.array([])

	if code.size % 2 == 1: # add NULL bit if ODD LENGTH
		code = np.append(code, 0)
	
	code = np.reshape(code, (2, int(code.size / 2)) )	
	SymbolLen = int(SamplePerBit * code.size / 2)

	for SampleNum in range(SymbolLen):
		Index = int(SampleNum/SamplePerBit)
		InphaseBitSequence = np.insert(InphaseBitSequence, SampleNum, code[0,Index])
		QuadBitSequence = np.insert(QuadBitSequence, SampleNum, code[1,Index])

	BitSequence = np.array([InphaseBitSequence, QuadBitSequence])
	#bit_seq_I = np.append(BitSequence, code[int(sample / n_sample_bit)])
	#bit_seq_Q = np.append(bit_seq_Q, code[int(np.ceil(code.size/2) + sample / n_sample_bit)])
	return BitSequence


if __name__ == "__main__":

	# Parameter and code set-up
	fc = 48 * pow(10,3)
	fs = 1.25 * pow(10,6)
	Ts = 1/fs
	Tc = 1/fc

	# barker-4
	#code = np.array([1, 1, -1, 1])
	#code_hex = 0x0D
# barker-5
	# code = np.array([1, 1, 1, -1, 1])
	#CodeHex = 0x1D
# barker-7
	# code = np.array([1, 1, 1, -1, -1, 1, -1]) 
	CodeHex = 0x72
# barker-13
	# code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]) 
	# code_hex = 0x1F35
# orthogonal-16bit
	# code = np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1])
	#code_hex = 0x3E25
# kasami-15
	#code = np.array([1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1])
	#CodeHex = 0x44D7
# gold-31
	#code = np.array([-1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, \
	# 	1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])
	# CodeHex = 0x766B8C7F

# kasami-15 double
	#code = np.array([1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1])
	#CodeHex = 0x44D7 + 0x44D7 * pow(2,15)
	

	CodeSize = 7
	Code = hex2bin(CodeHex, CodeSize)	# store as bipolar

	print("o Code (", str(CodeSize), "length)")
	print("- Hex: ", "0x%x" % CodeHex)
	print("- Binary: ", (Code + 1)/2)	# print as unipolar
	print("") 


	nPulsePerBit = 12
	nPulseSample = fs / fc
	nBitSample = nPulseSample * nPulsePerBit
	nCodeSample = nBitSample * Code.size
	print("o Pulse needed")
	print("- Pulse: ", "%.3f" % nPulseSample, "(samples), ", "%.3f" % (nPulseSample * Ts * 1000), "(ms)")
	print("- 1Bit: ", "%.3f" % nBitSample, "(samples), ", "%.3f" % (nBitSample * Ts * 1000), "(ms)")
	print("- Code: ", "%.3f" % nCodeSample, "(samples), ", "%.3f" % (nCodeSample * Ts * 1000), "(ms)")
	print("")
	

	# Generate Code Sequence and TX Data
	ModData = DQPSKmodulation(Code, Tc, Ts, nPulsePerBit)
	SymbolSequence = generateCodeStream(Code, nBitSample)
	SymbolSequenceSize = np.size(SymbolSequence,1)

	print("o Symbol sequence size")
	print("- In-phase: ", SymbolSequenceSize)
	print("- Quadrature: ", SymbolSequenceSize)
	print("")

	# TX data to Transducer
	# - Time Parameter (Delay, Simulation Time)
	MaxTime = 11 * 2 / 340
	MinTime = 0.3 * 2 / 340
	Distance= 2	# (m)
	SetDelay = Distance * 2 / 340
	nRTTSample = int(SetDelay / Ts)
	#RTT = np.random.random_sample() * (MaxTime - MinTime - nCodeSample*Ts) + MinTime
	#nRTTSample = int(RTT / Ts)
	nMaxTimeSample = int(MaxTime / Ts)
	nRestSample = int(nMaxTimeSample - nRTTSample - nCodeSample)

	# Generate TX / RX signal 
	Amplitude = 12
	EbN0dB = np.arange(-10,51,1)
	SNR = pow(10,EbN0dB/10)
	N0 = np.sqrt(2) * pow(Amplitude, 2) / SNR
	AttenuationCoef = -1.46017

	# -> Make TX Signal
	TrxSignal = np.append(np.zeros(nRTTSample), ModData)
	TrxSignal = np.append(TrxSignal, np.zeros(nRestSample)) * Amplitude
	TrxSignal = butter_bandpass_filter(TrxSignal, fc - 2000, fc + 2000, fs, 1)
	
	# -> Insert channel property
	Attenuated = TrxSignal * pow(10,AttenuationCoef/20 * nRTTSample * Ts * 340)
	Noise = np.random.normal(0, 1, Attenuated.size) * np.sqrt(N0[10]/2)
	
	# -> Make Rcv Signal
	RcvSignal = Attenuated + Noise
	RcvSignal = butter_bandpass_filter(RcvSignal, fc - 2000, fc + 2000, fs, 1)

	# -> Set time axis 
	tSignal = np.arange(RcvSignal.size) * Ts * 1000
	tSymbol = (np.arange(0, SymbolSequenceSize, 1) + nRTTSample) * Ts * 1000

	# Demodulation
	# -> 1.5m -> 11029, 1m -> 7352, 30cm -> 2205 in 48kHz
	RcvStartingSample = SymbolSequenceSize
	print("o Demodulation starts from sample", RcvStartingSample)
	print("")

	tDemod = tSignal[RcvStartingSample:-int(nBitSample)]	# in ms
	InphaseDemod = RcvSignal[RcvStartingSample+int(nBitSample):] 
	InphaseDemod = InphaseDemod * RcvSignal[RcvStartingSample:-int(nBitSample)]
	QuadDemod = RcvSignal[RcvStartingSample+int(nBitSample)-int(nPulseSample/4):-int(nPulseSample/4)]  
	QuadDemod = QuadDemod * RcvSignal[RcvStartingSample:-int(nBitSample)]
	
	InphaseEnvelop = butter_lowpass_filter(InphaseDemod, 3*pow(10,3), fs, 4) / InphaseDemod.size
	QuadEnvelop = butter_lowpass_filter(QuadDemod, 3*pow(10,3), fs, 4) / QuadDemod.size

	print("o Envelope size")
	print("- In-phase: ", InphaseEnvelop.size)
	print("- Quadrature: ", QuadEnvelop.size)
	print("")


	# Correlation
	InphaseCorrelation = signal.correlate(InphaseEnvelop, SymbolSequence[(0,slice(None))], mode='valid')
	QuadCorrelation = signal.correlate(QuadEnvelop, SymbolSequence[(1,slice(None))], mode='valid') 
	Correlation = (InphaseCorrelation + QuadCorrelation)
	Correlation = Correlation / Correlation.size

	tCorrelation = np.arange(RcvStartingSample, \
				(RcvStartingSample+Correlation.size), 1) * Ts * 1000 # in ms
	CorrelationPeak = np.argmax(Correlation)

	print("o Delay: ", nRTTSample, "/ %.2f" % (nRTTSample * Ts * 1000), "ms")
	print("o Distance: ", "%.2f" % (nRTTSample * Ts * 340 / 2), "m")
	print("o Correlation peak: ", CorrelationPeak, "/ %.2f" % (CorrelationPeak * Ts * 1000), "ms")
	print("o Detected distance: ", "%.2f" % \
				((RcvStartingSample + CorrelationPeak) * Ts * 340 / 2), "m")


	# Plot
	fig, axes = plt.subplots(2,3, sharex=True)
	
	axes[0,0].plot(tSymbol, SymbolSequence[0,:], 'b-')
	axes[0,0].set_title('Code Sequence')
	axes[0,0].set_xlabel('Time (ms)')
	axes[0,0].set_ylabel('Inphase', color='b')
	axes[0,0].set_ylim(-3.5,1.2)
	axes2 = axes[0,0].twinx()
	axes2.plot(tSymbol, SymbolSequence[1,:], 'm-')
	axes2.set_ylabel('Quadrature', color='m')
	axes2.set_ylim(-1.2,3.5)

	axes[0,1].plot(tSignal, TrxSignal)
	axes[0,1].set_xlabel('Time (ms)')
	axes[0,1].set_title('Transmitted Data')
	
	axes[0,2].plot(tSignal, RcvSignal)
	axes[0,2].set_xlabel('Time (ms)')
	axes[0,2].set_ylabel('Amplitude')
	axes[0,2].set_title('Received Data')
	
	axes[1,0].plot(tDemod, InphaseEnvelop, 'b-')
	axes[1,0].set_xlabel('Time (ms)')
	axes[1,0].set_ylabel('Demodulation', color='b')
	axes[1,0].set_title('I-Demodulation')
	axes2 = axes[1,0].twinx()
	axes2.plot(tSymbol, SymbolSequence[0,:], 'm-')
	axes2.set_ylabel('Code', color='m')
	
	axes[1,1].plot(tDemod, QuadEnvelop, 'b-')
	axes[1,1].set_xlabel('Time (ms)')
	axes[1,1].set_ylabel('Demodulation', color='b')
	axes[1,1].set_title('Q-Demodulation')
	axes2 = axes[1,1].twinx()
	axes2.plot(tSymbol, SymbolSequence[1,:], 'm-')
	axes2.set_ylabel('Code', color='m')
	
	axes[1,2].plot(tCorrelation, Correlation)
	axes[1,2].set_xlabel('Time (ms)')
	axes[1,2].set_title('Correlation')

	plt.show()