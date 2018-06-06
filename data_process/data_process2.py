import os
import sys
from copy import deepcopy
import time
import string

dataDir = 'Dataset_Alg_SourceDevice'
fileList = os.listdir(dataDir)

dotList = []
for fileName in fileList:
	if fileName[0] == '.':
		dotList.append(fileName)

for elem in dotList:
	fileList.remove(elem)

# pairList = []
# singleList = []
# while len(fileList) > 1:
# 	curA = fileList[0]
# 	curPair = [curA]
# 	nameElem = curA.split('-')
# 	curLabel = '-'.join(nameElem[:-1])
# 	for idx in xrange(1,len(fileList)):
# 		if curLabel in fileList[idx]:
# 			curPair.append(fileList[idx])
# 	if len(curPair) == 2:
# 		pairList.append(deepcopy(curPair))
# 		for pairElem in curPair:
# 			fileList.remove(pairElem)
# 	elif len(curPair) == 1:
# 		# print curLabel
# 		fileList = fileList[1:]
# 		singleList.append(curPair[0])
# 	else:
# 		print curLabel
# 		print 'Wrong', len(curPair), curPair
# 		fileList = fileList[1:]
# 		fileList.append(curPair[0])

timeLabel = 'Creation_Time'
pairSaveDir = 'Dataset_AccGry_SourceDevice'+'-'+timeLabel+'-avgTime'


if not os.path.exists(pairSaveDir):
	os.mkdir(pairSaveDir)

for fileName in fileList:
	print fileName
	pairFile1 = []
	fileIn1 = open(os.path.join(dataDir, fileName))
	line = fileIn1.readline()
	count = 0
	while len(line) > 0:
		curElem = eval(line)
		curElem['MagnetData0'] = eval(curElem['MagnetData0'])
		curElem['MagnetData1'] = eval(curElem['MagnetData1'])
		curElem['MagnetData2'] = eval(curElem['MagnetData2'])
		curElem['LightData'] = eval(curElem['LightData'])
		curElem['PressureData'] = eval(curElem['PressureData'])
		curElem['GpsNumber'] = eval(curElem['GpsNumber'])
		mytime = str(curElem['Time'])
		msElem = string.atof(mytime[14:17])/1000
		mytime = mytime[0:4] + '-' + mytime[4:6] + '-' + mytime[6:8] + ' ' + mytime[8:10] + ':' + mytime[10:12] + ':' + mytime[12:14]
		timeArray = time.strptime(mytime, "%Y-%m-%d %H:%M:%S")
		curElem['Time'] = time.mktime(timeArray) + msElem


		curElem['GpsSNR'] = eval(curElem['TotalSnr'])


		# curElem['GpsInfo'] = 0
		curElem['0snr15'] = 0
		curElem['15snr30'] = 0
		curElem['30snr50'] = 0

		if curElem['GpsNumber'] != 0:
			snr = curElem['GpsInfo']
			gpselem = snr.split('#')
			# print gpselem
			curgps = []

			for gps in gpselem:
				snrelem = gps.split('|')
				# print snrelem
				curgps.append(snrelem[0])

			for s in curgps:
				if eval(s) < 15:
					curElem['0snr15'] += 1
				elif 15 <= eval(s) < 30:
					curElem['15snr30'] += 1
				else:
					curElem['30snr50'] += 1

		pairFile1.append(deepcopy(curElem))
		line = fileIn1.readline()
		count += 1
		print '\r', count,
		sys.stdout.flush()
	print ''
	fileIn1.close()


	idx1 = 0
	fileLable = fileName.split('.')[0]
	curLabel = '-'.join(['data', fileLable])
	fileOut = open(os.path.join(pairSaveDir, curLabel),'w')
	print 'Write', curLabel
	count = 0
	while idx1 < len(pairFile1):
		curItem1 = pairFile1[idx1]

		# print curTime1, curTime2, curTime1==curTime2
		curSaveElem = {}

		curSaveElem['Magnetic_field'] = deepcopy([curItem1['MagnetData0'], curItem1['MagnetData1'], curItem1['MagnetData2']])
		curSaveElem['LightData'] = deepcopy([curItem1['LightData']])
		curSaveElem['PressureData'] = deepcopy([curItem1['PressureData']])
		curSaveElem['GPS'] = deepcopy([curItem1['GpsNumber'], curItem1['GpsSNR'], curItem1['0snr15'], curItem1['15snr30'], curItem1['30snr50']])
		curSaveElem['Time'] = deepcopy(curItem1['Time'])

		fileOut.write(str(curSaveElem)+'\n')
		idx1 += 1
		count += 1
		print '\r', count,
		sys.stdout.flush()

	print ''
	fileOut.close()

