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


def generatePulseStream(code, SamplePerBit):
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
	#CodeHex = 0x766B8C7F
	
	CodeSize = 7
	Code = hex2bin(CodeHex, CodeSize)	# store as bipolar

	print("o Code ("+ str(CodeSize), "length)")
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

	ModData = DQPSKmodulation(Code, Tc, Ts, nPulsePerBit)
	SymbolSequence = generatePulseStream(Code, nBitSample)
	SymbolSequenceSize = np.size(SymbolSequence,1)

	tSymbol = np.arange(0, Ts * (SymbolSequenceSize),Ts) * 1000 # in ms
	
	print("o Symbol sequence size")
	print("- In-phase: ", SymbolSequenceSize)
	print("- Quadrature: ", SymbolSequenceSize)
	print("")

	MaxTime = 3 * 2 / 340
	MinTime = 0.3 * 2 / 340
	RTT = np.random.random_sample() * (MaxTime - MinTime - nCodeSample*Ts) + MinTime
	nRTTSample = int(RTT / Ts)
	nMaxTimeSample = int(MaxTime / Ts)
	nRestSample = int(nMaxTimeSample - nRTTSample - nCodeSample)

	TrxSignal = np.append(np.zeros(nRTTSample), ModData)
	TrxSignal = np.append(TrxSignal, np.zeros(nRestSample))

	Attenuated = TrxSignal
	RcvSignal = Attenuated + np.random.normal(0,0.5,Attenuated.size)

	# Demodulation
	  # 1.5m -> 11029, 1m -> 7352, 30cm -> 2205 in 48kHz
	RcvStartingSample = SymbolSequenceSize
	print("o Demodulation starts from sample", RcvStartingSample)
	print("")

	tDemod = tSymbol[RcvStartingSample:-int(nBitSample)]	# in ms
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

	plt.plot(InphaseDemod)
	plt.show()