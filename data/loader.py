import os
import sys
import csv
import torch
from PIL import Image

def load_train_data_multi(xFolder_list, trainLogPath_list):
	## prepare for getting x
	for xFolder in xFolder_list:
		if not os.path.exists(xFolder):
			sys.exit('Error: the image folder is missing. ' + xFolder)
		
	## prepare for getting y
	trainLog_list = []
	for trainLogPath in trainLogPath_list:
		if not os.path.exists(trainLogPath):
			sys.exit('Error: the labels.csv is missing. ' + trainLogPath)
		with open(trainLogPath, newline='') as f:
			trainLog = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
			trainLog_list.append(trainLog)

	## get x and y
	xList, yList = ([], [])
	i = 0
	for trainLog in trainLog_list:
		xFolder = xFolder_list[i]
		xList_1 = []
		yList_1 = []
		for row in trainLog:
			## center camera
			xList_1.append(os.path.join(xFolder, os.path.basename(row[0]))) 
			yList_1.append(float(row[3]))     
		xList = xList + xList_1
		yList = yList + yList_1
		i+=1
	return (xList, yList)

class DatasetfromList(torch.utils.data.Dataset):

	def __init__(self, x_list, y_list, transform=None):
		assert(len(x_list) == len(y_list))
		self.imageList = x_list
		self.labelList = y_list
		self.transform = transform
	
	def __len__(self):
		return(len(self.imageList))
	
	def __getitem__(self, idx):
		img_path = self.imageList[idx]
		img = Image.open(img_path)
		if self.transform is not None:
			img = self.transform(img)

		label = self.labelList[idx]
		label = torch.tensor(label).float()

		return img, label
