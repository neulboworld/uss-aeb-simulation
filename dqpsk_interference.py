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
		temp = generatePulse(p, Tc, Ts, nPulsePerBit)
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

	# Generate TX / RX signal 
	Amplitude = 12
	EbN0dB = np.arange(-10,51,1)
	SNR = pow(10,EbN0dB/10)
	N0 = np.sqrt(2) * pow(Amplitude,2) / SNR
	AttenuationCoef = -1.46017

	ErrCntArray = np.array([])
	DistArray = np.array([])
	PeakArray = np.array([])

	Dist = np.append([0.5],range(1,10,1))

	for dist in np.array([2]):#Dist:
		
		ErrCnt = 0
		print(dist)

		for itr in range(10):

			if dist > 1:
				CodeHex = 0x766B8C7F
				CodeSize = 31
			else:
				CodeHex = 0x72
				CodeSize = 7

			Code = hex2bin(CodeHex, CodeSize)	# store as bipolar
			CodeHex2 = np.random.randint(0, CodeSize)
			Code2 = hex2bin(CodeHex2 ,CodeSize)

			nPulsePerBit = 4
			nPulseSample = fs / fc
			nBitSample = nPulseSample * nPulsePerBit
			nCodeSample = nBitSample * Code.size

			# Generate Code Sequence and TX Data
			ModData = DQPSKmodulation(Code, Tc, Ts, nPulsePerBit)
			ModData2 = DQPSKmodulation(Code2, Tc, Ts, nPulsePerBit)
			SymbolSequence = generateCodeStream(Code, nBitSample)
			SymbolSequenceSize = np.size(SymbolSequence,1)

			# TX data to Transducer
			# - Time Parameter (Delay, Simulation Time)
			MaxTime = 12 * 2 / 340
			MinTime = 0.3 * 2 / 340
			Distance = dist 	# (m)
			SetDelay = Distance * 2 / 340
			SetDelay = np.append(SetDelay, np.random.rand() * (MaxTime - MinTime - nCodeSample*Ts) + MinTime)
			nRTTSample = (SetDelay / Ts).astype(int) 
			#RTT = np.random.rand(2) * (MaxTime - MinTime - nCodeSample*Ts) + MinTime
			#nRTTSample = (RTT / Ts).astype(int)
			nMaxTimeSample = int(MaxTime / Ts)
			nRestSample = (nMaxTimeSample - nRTTSample - nCodeSample).astype(int)


			# -> Make Data Signal
			DataSignal = np.append(np.zeros(nRTTSample[0]), ModData)
			DataSignal = np.append(DataSignal, np.zeros(nRestSample[0])) * Amplitude

			
			DataSignal = butter_bandpass_filter(DataSignal, fc - 1200, fc + 1200, fs, 1)	
			Attenuated = DataSignal* pow(10,AttenuationCoef/20 * nRTTSample[0] * Ts * 340)

			# -> Make Interference
			Interference = np.append(np.zeros(nRTTSample[1]), ModData2)
			Interference = np.append(Interference, np.zeros(nRestSample[1])) * Amplitude
			Interference = butter_bandpass_filter(Interference, fc - 1200, fc + 1200, fs, 1)
			Interference = Interference * np.random.rand() * (pow(10,AttenuationCoef/20 * 0.3) + 0.2)

			# -> Make Tx Signal
			TrxSignal = Attenuated + Interference

			# -> Make Rcv Signal

			Noise = np.random.normal(0, 1, TrxSignal.size) * np.sqrt(N0[31]/2)
			RcvSignal = Interference + Noise
			RcvSignal = butter_bandpass_filter(RcvSignal, fc - 2000, fc + 2000, fs, 1)

			# -> Set time axis 
			tSignal = np.arange(RcvSignal.size) * Ts * 1000
			tSymbol = (np.arange(0, SymbolSequenceSize, 1) + nRTTSample[0]) * Ts * 1000
			tSymbol2 = (np.arange(0, SymbolSequenceSize, 1) + nRTTSample[1]) * Ts * 1000

			# Demodulation
			# -> 1.5m -> 11029, 1m -> 7352, 30cm -> 2205 in 48kHz
			RcvStartingSample = SymbolSequenceSize
			'''
			print("o Demodulation starts from sample", RcvStartingSample)
			print("")
			'''

			tDemod = tSignal[RcvStartingSample:-int(nBitSample)]	# in ms
			InphaseDemod = RcvSignal[RcvStartingSample+int(nBitSample):] 
			InphaseDemod = InphaseDemod * np.real(signal.hilbert(InphaseDemod))#RcvSignal[RcvStartingSample:-int(nBitSample)]
			QuadDemod = RcvSignal[RcvStartingSample+int(nBitSample)-int(nPulseSample/4):-int(nPulseSample/4)]  
			QuadDemod = QuadDemod * np.real(signal.hilbert(QuadDemod)) #RcvSignal[RcvStartingSample:-int(nBitSample)]
			
			InphaseEnvelop = butter_lowpass_filter(InphaseDemod, 3*pow(10,3), fs, 4) / InphaseDemod.size
			QuadEnvelop = butter_lowpass_filter(QuadDemod, 3*pow(10,3), fs, 4) / QuadDemod.size
			'''
			print("o Envelope size")
			print("- In-phase: ", InphaseEnvelop.size)
			print("- Quadrature: ", QuadEnvelop.size)
			print("")
			'''

			# Correlation
			InphaseCorrelation = signal.correlate(InphaseEnvelop, SymbolSequence[(0,slice(None))], mode='valid')
			QuadCorrelation = signal.correlate(QuadEnvelop, SymbolSequence[(1,slice(None))], mode='valid') 
			Correlation = (InphaseCorrelation + QuadCorrelation)
			#Correlation = Correlation / Correlation.size

			tCorrelation = np.arange(RcvStartingSample, \
						(RcvStartingSample+Correlation.size) , 1) * Ts  * 1000 # in ms
			CorrelationPeak = np.argmax(Correlation)
			
			Estimation = (RcvStartingSample + CorrelationPeak) * Ts * 340 / 2 - 0.05

			if abs(Estimation - Distance > 0.2):
				ErrCnt = ErrCnt + 1
			PeakArray = np.append(PeakArray, Correlation[CorrelationPeak])
			DistArray = np.append(DistArray, Estimation)
		
		DistArray = np.reshape(DistArray, (int(DistArray.size/10),10) )
		PeakArray = np.reshape(PeakArray, (int(PeakArray.size/10),10) )
		ErrCntArray = np.append(ErrCntArray, ErrCnt)
		print("Error Count:", ErrCnt)
	
	print(DistArray)
	print(PeakArray)
	print(ErrCntArray)
	'''
	print("o Delay: ", nRTTSample[0], "/ %.2f" % (nRTTSample[0] * Ts * 1000), "ms")
	print("o Distance: ", "%.2f" % (nRTTSample[0] * Ts * 340 / 2), "m")
	print("o Correlation peak: ", CorrelationPeak, "/ %.2f" % (CorrelationPeak * Ts * 1000), "ms")
	print("o Detected distance: ", "%.2f" % \
				((RcvStartingSample + CorrelationPeak) * Ts * 340 / 2), "m")
	'''
	
	# Plot
	fig, axes = plt.subplots(2,3, sharex=True)
	axes[0,0].plot(tSymbol, SymbolSequence[0,:], 'b-')
	axes[0,0].set_title('Code Sequence')
	axes[0,0].set_xlabel('time')
	axes[0,0].set_ylabel('Inphase', color='b')
	axes[0,0].set_ylim(-3.5,1.2)
	axes2 = axes[0,0].twinx()
	axes2.plot(tSymbol, SymbolSequence[1,:], 'm-')
	axes2.set_ylabel('Quadrature', color='m')
	axes2.set_ylim(-1.2,3.5)

	axes[0,1].plot(tSignal, Attenuated, tSignal, Interference)
	axes[0,1].set_xlabel('time')
	axes[0,1].set_title('Transmitted Data')
	
	axes[0,2].plot(tSignal, RcvSignal)
	axes[0,2].set_xlabel('time')
	axes[0,2].set_title('Received Data')
	
	axes[1,0].plot(tDemod, InphaseEnvelop, 'b-')
	axes[1,0].set_xlabel('time')
	axes[1,0].set_ylabel('Demodulation', color='b')
	axes[1,0].set_title('I-Demodulation')
	axes2 = axes[1,0].twinx()
	axes2.plot(tSymbol, SymbolSequence[0,:], 'm-')
	axes2.set_ylabel('Code', color='m')
	
	axes[1,1].plot(tDemod, QuadEnvelop, 'b-')
	axes[1,1].set_xlabel('time')
	axes[1,1].set_ylabel('Demodulation', color='b')
	axes[1,1].set_title('Q-Demodulation')
	axes2 = axes[1,1].twinx()
	axes2.plot(tSymbol, SymbolSequence[1,:], 'm-')
	axes2.set_ylabel('Code', color='m')

	axes[1,2].plot(tCorrelation, Correlation)
	axes[1,2].set_xlabel('time')
	
	'''
	fig = plt.plot(Dist,ErrCntArray/100)
	plt.xlabel("Distance (m)")
	plt.ylabel("Error Rate (%)")
	plt.grid()
	'''
	plt.show()
	
