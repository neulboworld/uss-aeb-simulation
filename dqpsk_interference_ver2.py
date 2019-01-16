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

def hex2bin(hexArray,size):

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

def DQPSKmodulation(code, size, Tc, TS, nPulse):
	
	BitArray = np.array([])
	for i in range(size):
		BitArray = np.append(BitArray, ((code & pow(2,i)) >> i) * 2 -1)

	BitArray = np.flip(BitArray)

	ModulatedPulse = np.array([])

	if BitArray.size % 2 == 1: # add NULL bit if ODD LENGTH
		BitArray = np.append(BitArray, 0)

	BitArray = np.reshape(BitArray, (2, int(BitArray.size / 2)) ) 
	phaseArray = np.array([0])
	phase = 0

	for index in range(np.size(BitArray,1)):
		if BitArray[(0,index)] == 1:
			if BitArray[(1,index)] == 1:
				phase = phase + 45
			elif BitArray[(1,index)] == -1:
				phase = phase + 315

		elif BitArray[(0,index)] == -1:
			if BitArray[(1,index)] == 1:
				phase = phase + 135
			elif BitArray[(1,index)] == -1:
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


def generateCodeStream(code, size, SamplePerBit):
	InphaseBitSequence = np.array([])
	QuadBitSequence = np.array([])

	BitArray = np.array([])
	for i in range(size):
		BitArray = np.append(BitArray, ((code & pow(2,i)) >> i) * 2 -1)

	BitArray = np.flip(BitArray)

	ModulatedPulse = np.array([])

	if BitArray.size % 2 == 1: # add NULL bit if ODD LENGTH
		BitArray = np.append(BitArray, 0)

	BitArray = np.reshape(BitArray, (2, int(BitArray.size / 2)) ) 

	if BitArray.size % 2 == 1: # add NULL bit if ODD LENGTH
		BitArray = np.append(BitArray, 0)
	
	BitArray = np.reshape(BitArray, (2, int(BitArray.size / 2)) )
	SymbolLen = int(SamplePerBit * BitArray.size / 2)

	for SampleNum in range(SymbolLen):
		Index = int(SampleNum/SamplePerBit)
		InphaseBitSequence = np.insert(InphaseBitSequence, SampleNum, BitArray[0,Index])
		QuadBitSequence = np.insert(QuadBitSequence, SampleNum, BitArray[1,Index])

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
	Dist = np.append([0.3],range(1,10,1))

	for dist in np.array([2]):
		
		ErrCnt = 0
		print(dist)

		for itr in range(1):

			Code = np.array([0x766B8C7F, 0x4259F1BA, 0x4327DC56, 0x017E2DEC])
			CodeSize = 31

			# Code = np.array([0x410C53D1C96ECD5F, 0x12AB1D4CF31A248C, 0x6642CEEBBD871EF8, 0x0F9169A520BD6A1])
			# CodeSize = 63
			'''
			'410C53D1C96ECD5F'
		    '12AB1D4CF31A248C'
		    '6642CEEBBD871EF8'
		    '0F9169A520BD6A11'
		    '5C3627381AC983C2'
		    '7B78BA026E205065'
		    '35E5807687F3F72B'
		    '28DFF49F5454B9B6'

			'''
			'''
			'4259F1BA'
		    '4327DC56'
		    '017E2DEC'
		    '44164917'
		    '4EC680E0'
		    '5B67130E'
		    '702434D2'
		    '26A27B6A'
		    '0BAEE41B'
		    '51B7DAF9'
		    '6585A73C'
		    '0DE15CB6'
		    '5D28ABA3'
		    '7CBB4588'
		    '3F9C99DE'
		    '39D32173'
		    '354C5029'
		    '2C72B29D'
		    '1E0F77F5'
		    '7AF4FD25'
		    '3303E884'
		    '20EDC3C7'
		    '07319541'
		    '4889384D'
		    '57F86254'
		    '691AD666'
		    '14DFBE02'
		    '6F556ECB'
		    '1840CF58'
		    '766B8C7F'
		    '2A3D0A30'
		    '129006AF'
		    '63CA1F91'
			#CodeHex = 0x72
			#CodeSize = 7
			'''

			nPulsePerBit = 12
			nPulseSample = fs / fc
			nBitSample = nPulseSample * nPulsePerBit
			nCodeSample = nBitSample * CodeSize

			print("o Pulse needed")
			print("- Pulse: ", "%.3f" % nPulseSample, "(samples), ", "%.3f" % (nPulseSample * Ts * 1000), "(ms)")
			print("- 1Bit: ", "%.3f" % nBitSample, "(samples), ", "%.3f" % (nBitSample * Ts * 1000), "(ms)")
			print("- Code: ", "%.3f" % nCodeSample, "(samples), ", "%.3f" % (nCodeSample * Ts * 1000), "(ms)")
			print("")

			# Generate Code Sequence and TX Data
			ModData = np.array([])
			
			for i in range(4):
				ModData = np.append(ModData, \
						DQPSKmodulation(Code[i], CodeSize, Tc, Ts, nPulsePerBit) )

			ModData = np.reshape(ModData, (4, int(ModData.size/4)) )
			
			SymbolMatrix_I = np.array([])
			SymbolMatrix_Q = np.array([])

			for i in range(4):
				SymbolSequence = generateCodeStream(Code[i], CodeSize, nBitSample)
				SymbolMatrix_I = np.append(SymbolMatrix_I, SymbolSequence[0,:])
				SymbolMatrix_Q = np.append(SymbolMatrix_Q, SymbolSequence[1,:])

			SymbolMatrix_I = np.reshape(SymbolMatrix_I, (4, int(SymbolMatrix_I.size/4)) )
			SymbolMatrix_Q = np.reshape(SymbolMatrix_Q, (4, int(SymbolMatrix_Q.size/4)))

			# TX data to Transducer
			# - Time Parameter (Delay, Simulation Time)
			DistMin = 0.3
			DistMax = 12
			MaxTime = DistMax * 2 / 340
			MinTime = DistMin * 2 / 340
			SensorPos = np.array([[-0.675, 0],[-0.225, 0],[0.225, 0],[0.675, 0]])
			ObstaclePos = np.random.rand(2) * np.array([9, 4.2]) - np.array([4.5, -0.3])
			#print("Obstacle Position:", ObstaclePos, "m")
			DistancePOW = (SensorPos - ObstaclePos)**2
			Distance = np.sqrt(DistancePOW[:,0] + DistancePOW[:,1])
			print("Distance:", Distance, "m")
			SetDelay = Distance * 2 / 340
			nRTTSample = (SetDelay / Ts).astype(int)
			nMaxTimeSample = int(MaxTime / Ts)
			nRestSample = (nMaxTimeSample - nRTTSample - nCodeSample).astype(int)

			TXMatrix = np.array([])

			# -> Make TX Signal
			for i in range(4):
				USSStream = np.append(np.zeros(nRTTSample[i]), ModData)
				USSStream = np.append(USSStream, np.zeros(nRestSample[i]) ) * Amplitude
				TXMatrix = np.append(TXMatrix, USSStream)

			TXMatrix = np.reshape(TXMatrix, (4, USSStream.size) )
			TXMatrix = butter_bandpass_filter(TXMatrix, fc - 1200, fc + 1200, fs, 1)	
			Attenuation = pow(10, AttenuationCoef/20 * \
						np.reshape(nRTTSample, (nRTTSample.size, 1)) * Ts * 340)
			TXMatrix = TXMatrix * Attenuation

			# -> Make RX Signal
			Noise = np.random.normal(0, 1, TXMatrix.shape) * np.sqrt(N0[41]/2)
			RXMatrix = TXMatrix + Noise
			RXSignal = RXMatrix[0,:] + 0.5*RXMatrix[1,:] #+ RXMatrix[2,:] + RXMatrix[3,:]
			RXSignal = butter_bandpass_filter(RXSignal, fc - 2000, fc + 2000, fs, 1)

			# -> Set time axis 
			tSignal = np.arange(RXSignal.size) * Ts * 1000	 # in ms

			# Demodulation
			# -> 1.5m -> 11029, 1m -> 7352, 30cm -> 2205 in 48kHz
			RcvStartingSample = int(nCodeSample)
			'''
			print("o Demodulation starts from sample", RcvStartingSample)
			print("")
			'''
			# -> Set time axis on demodulation
			tDemod = tSignal[RcvStartingSample:-int(nBitSample)]	# in ms
			
			# -> I-Demodulation
			InphaseDemod = RXSignal[(RcvStartingSample+int(nBitSample)):] 

			InphaseDemod = InphaseDemod * np.real(signal.hilbert(InphaseDemod))#RXSignal[RcvStartingSample:-int(nBitSample)]
			InphaseEnvelop = butter_lowpass_filter(InphaseDemod, 3*pow(10,3), fs, 4) / InphaseDemod.size

			# -> Q-Demodulation
			QuadDemod = RXSignal[RcvStartingSample+int(nBitSample)-int(nPulseSample/4):-int(nPulseSample/4)]  
			QuadDemod = QuadDemod * np.real(signal.hilbert(QuadDemod))#RXSignal[RcvStartingSample:-int(nBitSample)]
			QuadEnvelop = butter_lowpass_filter(QuadDemod, 3*pow(10,3), fs, 4) / QuadDemod.size

			'''
			print("o Envelope size")
			print("- In-phase: ", InphaseEnvelop.size)
			print("- Quadrature: ", QuadEnvelop.size)
			print("")
			'''

			# a = signal.correlate(SymbolMatrix_I[0,:], SymbolMatrix_I[0,:])
			# b = signal.correlate(SymbolMatrix_Q[0,:], SymbolMatrix_Q[0,:])
			
			# #plt.plot(SymbolMatrix_I[0,:])
			# plt.plot(a+b)
			# plt.show()

			# Correlation
			InphaseCorrelation = signal.correlate(InphaseEnvelop, SymbolMatrix_I[0,:], mode='valid')
			QuadCorrelation = signal.correlate(QuadEnvelop, SymbolMatrix_Q[0,:], mode='valid') 
			Correlation = (InphaseCorrelation + QuadCorrelation)
			#Correlation = Correlation / Correlation.size

			tCorrelation = np.arange(RcvStartingSample, \
						(RcvStartingSample+Correlation.size) , 1) * Ts * 1000 # in ms
			CorrelationPeak = np.argmax(Correlation)
			
			Estimation = (RcvStartingSample + CorrelationPeak) * Ts * 340 / 2 - 0.05
			print(Estimation)
			
			axes = plt.subplot(231)
			plt.plot(tSignal, RXMatrix[0,:])
			plt.subplot(232, sharex = axes)
			plt.plot(tSignal, RXMatrix[1,:])
			plt.subplot(233, sharex = axes)
			plt.plot(tSignal, RXSignal)
			plt.subplot(234, sharex = axes)
			plt.plot(tSignal, RXMatrix[2,:])
			plt.subplot(235, sharex = axes)
			plt.plot(tSignal, RXMatrix[3,:])
			plt.subplot(236, sharex = axes)
			plt.plot(tCorrelation, Correlation)
			plt.show()
			
			if np.abs(Estimation - Distance[0]) > 0.2:
				ErrCnt = ErrCnt + 1
	
		ErrCntArray = np.append(ErrCntArray, ErrCnt)
		print("Error Count:", ErrCnt)

	print(ErrCntArray)
	'''
	print("o Delay: ", nRTTSample[0], "/ %.2f" % (nRTTSample[0] * Ts * 1000), "ms")
	print("o Distance: ", "%.2f" % (nRTTSample[0] * Ts * 340 / 2), "m")
	print("o Correlation peak: ", CorrelationPeak, "/ %.2f" % (CorrelationPeak * Ts * 1000), "ms")
	print("o Detected distance: ", "%.2f" % \
				((RcvStartingSample + CorrelationPeak) * Ts * 340 / 2), "m")
	'''


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
	'''
	fig = plt.plot(Dist,ErrCntArray/100)
	plt.xlabel("Distance (m)")
	plt.ylabel("Error Rate (%)")
	plt.grid()
	'''
	#plt.show()
	
