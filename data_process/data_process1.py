import sys
import os
import numpy as np

# dataDir = os.path.join('Dataset','Activity recognition exp')
dataDir = 'environment_sensor'

fileInList = os.listdir(dataDir)

saveDir = 'Dataset_Alg_SourceDevice'
if not os.path.exists(saveDir):
	os.mkdir(saveDir)

for fileName in fileInList:
	if not '.csv' in fileName:
		continue

	fileIn = open(os.path.join(dataDir, fileName))
	line = fileIn.readline()
	print fileName
	headers = line[:-1].split(',')
	line = fileIn.readline()
	count = 0

	fileLable = fileName.split('.')[0]
	turn = int(fileLable)
	if 0 < turn < 1000:
		name = fileLable + "-indoor"
	elif 1000 < turn < 2000:
		name = fileLable + "-medium"
	else:
		name = fileLable + "-outdoor"
	while len(line) > 0:
		count += 1
		elems = line[:-1].split(',')
		curDict = dict(zip(headers, elems))
		# cur_User = curDict['User']
		# cur_Device = curDict['Device']
		# cur_gt = curDict['gt']

		subFileLable = '-'.join(['p', name] )  #xunlian wei p ceshi wei t
		fileOut = open(os.path.join(saveDir, subFileLable), 'a')
		fileOut.write(str(curDict)+'\n')
		fileOut.close()

		line = fileIn.readline()
		print '\r', count,
		sys.stdout.flush()
	print ''


