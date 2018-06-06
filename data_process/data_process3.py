import os
import sys
import numpy as np
from copy import deepcopy
from time import gmtime, strftime

from scipy.interpolate import interp1d
from scipy.fftpack import fft

timeLabel = 'Creation_Time'
pairSaveDir = 'Dataset_AccGry_SourceDevice' + '-' + timeLabel + '-avgTime'

mode = 'one_user_out'
select = 't'

curTime = gmtime()
curRunDir = strftime("%a-%d-%b-%Y-%H_%M_%S+0000", curTime)

# fileIn = open(os.path.join(pairSaveDir, '#DeivceBadData.csv'))
badDevice = ['lgwatch_1', 'lgwatch_2', 's3mini_2']
# for line in fileIn.readlines():
# 	if line[-1] == '\n':
# 		badDevice.append(line[:-1])
# 	else:
# 		badDevice.append(line)

dataList = os.listdir(pairSaveDir)
nameDevList = []
for dataFile in dataList:
	if dataFile[0] == '.':
		continue
	if dataFile[0] == '#':
		continue
	elems = dataFile.split('-')
	curLable = '-'.join(elems[:-1])
	if curLable not in nameDevList:
		nameDevList.append(curLable)
# print nameDevList

sepcturalSamples = 10
fftSpan =0.25
SampSpan = 20.
timeNoiseVar = 0.2
magNoiseVar = 0.5
gpsNoiseVar = 0.2
lightNoiseVar = 0.2
pressureNoiseVar = 0.2
augNum = 10

dataDict = {}
gtType = ["outdoor", "medium", "indoor"]
idxList = range(len(gtType))
gtIdxDict = dict(zip(gtType, idxList))
idxGtDict = dict(zip(idxList, gtType))
wide = 20
wideScaleFactor = 4
labEvery = False
for nameDev in nameDevList:
	for curGt in gtType:
		curType = gtIdxDict[curGt]
		curData_Sep = []
		if os.path.exists(os.path.join(pairSaveDir, nameDev + '-' + curGt)):
			fileIn = open(os.path.join(pairSaveDir, nameDev + '-' + curGt))
			line = fileIn.readline()
			curData = []
			curMagList = []
			curLightList = []
			curPressureList = []
			curGpsList = []
			curTimeList = []
			lastTime = -1
			startTime = -1
			count = 0
			print nameDev + '-' + curGt
			while len(line) > 0:
				curElem = eval(line)
				curTime = curElem['Time']
				if startTime < 0:
					startTime = curTime
					if startTime < 0:
						print 'Wrong!'
						test = raw_input('continue')
				if abs(curTime - startTime) > fftSpan:
					if len(curMagList) > 0:
						if curTimeList[-1] < fftSpan:
							curMagList.append(deepcopy(curMagList[-1]))
							curGpsList.append(deepcopy(curGpsList[-1]))
							curLightList.append(deepcopy(curLightList[-1]))
							curPressureList.append(deepcopy(curPressureList[-1]))
							curTimeList.append(fftSpan)
						curMagListOrg = np.array(curMagList).T  # turn
						curGpsListOrg = np.array(curGpsList).T
						curLightListOrg = np.array(curLightList).T
						curPressureListOrg = np.array(curPressureList).T
						curTimeListOrg = np.array(curTimeList)

						elems = nameDev.split('-')
						print '\r', count,
						sys.stdout.flush()

						if mode == 'one_user_out':  # ce shi shi bu jia zao sheng xunlian shi jia zaosheng(10 ci)
							if elems[1] == select:
								curAugNum = 1
							else:
								curAugNum = augNum
						elif mode == 'one_model_out':
							if select in nameDev[2:]:
								curAugNum = 1
							else:
								curAugNum = augNum
						else:
							curAugNum = augNum

						for augIdx in xrange(curAugNum):
							# augIdx = 0
							if augIdx == 0:
								curMagList = curMagListOrg + 0.
								curGpsList = curGpsListOrg + 0.
								curLightList = curLightListOrg + 0.
								curPressureList = curPressureListOrg + 0.
								curTimeList = curTimeListOrg + 0.
							else:
								curMagList = curMagListOrg + np.random.normal(0., magNoiseVar, curMagListOrg.shape)
								# curGpsList = curGpsListOrg + np.random.normal(0., gpsNoiseVar, curGpsListOrg.shape)  #### xuyaoxiugai
								curGpsList = curGpsListOrg
								curLightList = curLightListOrg + np.random.normal(0., lightNoiseVar, curLightListOrg.shape)
								curPressureList = curPressureListOrg + np.random.normal(0., pressureNoiseVar, curPressureListOrg.shape)
								curTimeList = curTimeListOrg + np.random.normal(0., timeNoiseVar, curTimeListOrg.shape)

							curTimeList = np.sort(curTimeList)
							if curTimeList[-1] < fftSpan:
								curTimeList[-1] = fftSpan
							if curTimeList[0] > 0.:
								curTimeList[0] = 0.

							magInterp = interp1d(curTimeList, curMagList)
							# print curAccList.shape
							# print curTimeList
							magInterpTime = np.linspace(0.0, fftSpan * 1, sepcturalSamples * 1)
							magInterpVal = magInterp(magInterpTime)
							magFFT = fft(magInterpVal).T

							magFFTSamp = magFFT[::1] / float(1)
							# print accFFTSamp.shape
							magFFTFin = []
							for magFFTElem in magFFTSamp:
								for axisElem in magFFTElem:
									magFFTFin.append(axisElem.real)  # shibu
									magFFTFin.append(axisElem.imag)  # fubu

							gpsInterp = interp1d(curTimeList, curGpsList)
							gpsInterpTime = np.linspace(0.0, fftSpan * 1, sepcturalSamples * 1)
							gpsInterpVal = gpsInterp(gpsInterpTime)
							gpsFFT = fft(gpsInterpVal).T
							gpsFFTSamp = gpsFFT[::1] / float(1)
							# print 'gyroFFTSamp', gyroFFTSamp.shape
							gpsFFTFin = []
							for gpsFFTElem in gpsFFTSamp:
								for axisElem in gpsFFTElem:
									gpsFFTFin.append(axisElem.real)
									gpsFFTFin.append(axisElem.imag)


							lightInterp = interp1d(curTimeList, curLightList)
							lightInterpTime = np.linspace(0.0, fftSpan * 1, sepcturalSamples * 1)
							lightInterpVal = lightInterp(lightInterpTime)
							lightFFT = fft(lightInterpVal).T
							lightFFTSamp = lightFFT[::1] / float(1)
							# print 'gyroFFTSamp', gyroFFTSamp.shape
							lightFFTFin = []
							for lightFFTElem in lightFFTSamp:
								for axisElem in lightFFTElem:
									lightFFTFin.append(axisElem.real)
									lightFFTFin.append(axisElem.imag)


							pressureInterp = interp1d(curTimeList, curPressureList)
							pressureInterpTime = np.linspace(0.0, fftSpan * 1, sepcturalSamples * 1)
							pressureInterpVal = pressureInterp(pressureInterpTime)
							pressureFFT = fft(pressureInterpVal).T
							pressureFFTSamp = pressureFFT[::1] / float(1)
							# print 'gyroFFTSamp', gyroFFTSamp.shape
							pressureFFTFin = []
							for pressureFFTElem in pressureFFTSamp:
								for axisElem in pressureFFTElem:
									pressureFFTFin.append(axisElem.real)
									pressureFFTFin.append(axisElem.imag)


							curSenData = []
							curSenData += magFFTFin
							curSenData += gpsFFTFin
							curSenData += lightFFTFin
							curSenData += pressureFFTFin        #(140)


							if len(curData) < augNum:
								curData.append([deepcopy(curSenData)])                  # jiaru zaosheng  zuo shujv robust
							# elif startTime - lastTime >= SampSpan and augIdx == 0:    #xunzhao lian xu de duanluo
								# for curAugData in curData:
								# 	curData_Sep.append(deepcopy(curAugData))
								# curData = []
								# curData.append([deepcopy(curSenData)])
							else:
								curData[augIdx].append(deepcopy(curSenData))    #meige 0.25s dou you 10 zu shujv (zaosheng )
						# for curAugData in curData:
						# 	curData_Sep.append(deepcopy(curAugData))
						# curData = []
						# curData.append([deepcopy(curSenData)])

						lastTime = startTime + curTimeList[-1]
						startTime = -1
						curMagList = []
						curGpsList = []
						curLightList = []
						curPressureList = []
						curTimeList = []
				if startTime < 0:
					startTime = curTime
					if startTime < 0:
						print 'Wrong!'
						test = raw_input('continue')
				if curTime - startTime not in curTimeList:
					curMagList.append(deepcopy(curElem['Magnetic_field']))
					curGpsList.append(deepcopy(curElem['GPS']))
					curLightList.append(deepcopy(curElem['LightData']))
					curPressureList.append(deepcopy(curElem['PressureData']))
					curTimeList.append(curTime - startTime)

				count += 1
				line = fileIn.readline()
		else:
			continue

		for curAugData in curData:
			curData_Sep.append(deepcopy(curAugData))

		# print 'curData_Sep', np.array(curData_Sep).shape
		if not dataDict.has_key(nameDev):
			dataDict[nameDev] = [[], []]

		for sepData in curData_Sep:
			staIdx = 0
			while staIdx < len(sepData):
				endIdx = min(staIdx + wide, len(sepData))
				if endIdx - staIdx < 5:
					# if endIdx - staIdx < wide:
					break
				dataDict[nameDev][0].append(deepcopy(sepData[staIdx:endIdx]))
				# curOut = [0] * len(gtType)
				# curOut[curType] = 1
				if labEvery:
					curOutPrep = []
					for outIdx in xrange(endIdx - staIdx):
						curOutPrep.append(deepcopy([curType]))
					dataDict[nameDev][1].append(deepcopy(curOutPrep))
				else:
					dataDict[nameDev][1].append(deepcopy([curType] ))
				staIdx += int(wide / wideScaleFactor)      #buchang  5 windows

X = []
Y = []
maskX = []
evalX = []
evalY = []
evalMaskX = []
paddingVal = 0.
inputFeature = sepcturalSamples * 10 * 2

count = 0
for nameDev in dataDict.keys():
	curX, curY = dataDict[nameDev]
	count += 1
	elems = nameDev.split('-')
	print '\r', count,
	sys.stdout.flush()
	if mode == 'one_user_out':
		if elems[1] == select:
			evalX += deepcopy(curX)
			evalY += deepcopy(curY)
			continue
	elif mode == 'one_model_out':
		if select in nameDev[1:]:
			evalX += deepcopy(curX)
			evalY += deepcopy(curY)
			continue
	X += deepcopy(curX)
	Y += deepcopy(curY)
#buquan shujv dao 20ge yizu  20  ge 5s window lianxu    20*140
for idx in xrange(len(X)):
	curLen = len(X[idx])
	maskX.append([[1.0]] * curLen)
	for addIdx in xrange(wide - curLen):
		X[idx].append([paddingVal] * inputFeature)
		maskX[idx].append([0.0])

for idx in xrange(len(evalX)):
	curLen = len(evalX[idx])
	evalMaskX.append([[1.0]] * curLen)
	for addIdx in xrange(wide - curLen):
		evalX[idx].append([paddingVal] * inputFeature)
		evalMaskX[idx].append([0.0])

X = np.array(X)
Y = np.array(Y)
maskX = np.array(maskX)

evalX = np.array(evalX)
evalY = np.array(evalY)
evalMaskX = np.array(evalMaskX)
print 'X', X.shape, X.dtype, 'Y', Y.shape, Y.dtype, 'maskX', maskX.shape, maskX.dtype
print 'evalX', evalX.shape, evalX.dtype, 'evalY', evalY.shape, evalY.dtype, 'evalMaskX', evalMaskX.shape, evalMaskX.dtype

X = np.reshape(X, [-1, wide * inputFeature])  #20*200 de yi hang data
XY = np.hstack((X, Y))                        #20*200+3(type) = 4803 data
print 'XY', XY.shape
evalX = np.reshape(evalX, [-1, wide * inputFeature])
evalXY = np.hstack((evalX, evalY))
print 'evalXY', evalXY.shape
out_dir = 'sepHARData_' + select
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
	os.mkdir(os.path.join(out_dir, 'train'))
	os.mkdir(os.path.join(out_dir, 'eval'))
idx = 0
for elem in XY:
	fileOut = open(os.path.join(out_dir, 'train', 'train_' + str(idx) + '.csv'), 'w')
	curOut = elem.tolist()
	curOut = [str(ele) for ele in curOut]
	curOut = ','.join(curOut) + '\n'
	fileOut.write(curOut)
	fileOut.close()
	idx += 1
idx = 0
for elem in evalXY:
	fileOut = open(os.path.join(out_dir, 'eval', 'eval_' + str(idx) + '.csv'), 'w')
	curOut = elem.tolist()
	curOut = [str(ele) for ele in curOut]
	curOut = ','.join(curOut) + '\n'
	fileOut.write(curOut)
	fileOut.close()
	idx += 1


