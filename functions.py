import cv2
import cv2.cv as cv
import numpy as np

def normalize(img):
	maxDarkValue = img.min()
	eyeNorm = img[:]
	imgSize = len(img),len(img[1])
	for i in range(imgSize[0]):
		for j in range(imgSize[1]):
			eyeNorm[i,j] = img[i,j] - maxDarkValue +1

	return eyeNorm

def percentage(part, whole):
  return 100 * float(part)/float(whole)


#Funzione che restituisce gli istogrammi, calcolati sull'asse delle x e delle y, 
#dove ogni elemento indica quanti punti hanno il colore (grigio)
#sotto una certa soglia. La soglia si autoadatta calcolando su quanti punti dell'asse x
#dell'asse x l'istogramma fa zero.

#This function returns the histograms, calculated on the x and y axis, where each element indicates how many
#points have the gray scale values under a certatin threshold. The threshold is adapted automatically calculating
#on how many points of the x axis the histhogram is 0.

def getHistograms(roiImg, offset, startThreshold):
	percentageThreshold = 55.0;
	threshold = startThreshold
	#Creazione dei vettori che conterranno i valori dell'istogramma
	xPupilAxis = [0] * (2*offset)
	yPupilAxis = [0] * (2*offset)

	for i in range(len(xPupilAxis)):
		for j in range(len(yPupilAxis)):
				#print (img[centerX+i-offset,centerY+j-offset])
				# roiImg[i,j] = roiImg[i,j]-maxDarkValue
				if roiImg[i,j] < threshold:
					xPupilAxis[j] = xPupilAxis[j]+1
					yPupilAxis[i] = yPupilAxis[i]+1
	xZeros = xPupilAxis.count(0)
	yZeros = yPupilAxis.count(0)
	while percentage(xZeros, len(xPupilAxis)) < percentageThreshold and threshold >= 0:
		xPupilAxis = [0] * (2*offset)
		yPupilAxis = [0] * (2*offset)

		threshold = threshold-1
		#Calcolo istogramma lungo l asse x e y
		for i in range(len(xPupilAxis)):
			for j in range(len(yPupilAxis)):
				#print (img[centerX+i-offset,centerY+j-offset])
				# roiImg[i,j] = roiImg[i,j]-maxDarkValue
				if roiImg[i,j] < threshold:
					xPupilAxis[j] = xPupilAxis[j]+1
					yPupilAxis[i] = yPupilAxis[i]+1
		xZeros = xPupilAxis.count(0)
		yZeros = yPupilAxis.count(0)

	return xPupilAxis, yPupilAxis

def getPupil2(img, centerX, centerY, radius):
	img_copy = np.copy(img)
	#Offset: distance from the center, from where we want to calculat the histogram
	offset = int(float(radius))-20
	threshold = 20

	isPupilCatched = True

	#Roi of pixels around the pupil
	roiY1 = int(centerY)-offset
	roiY2 = int(centerY)+offset
	roiX1 = int(centerX)-offset
	roiX2 = int(centerX)+offset
	if (roiX1>0 and roiX2>0 and roiY1>0 and roiY2>0):
		roiImg = img[roiY1:roiY2,roiX1:roiX2]
		cv2.imshow('pupilNotNorm',roiImg)
		# roiImg = normalize(roiImg)
		roiImg = cv2.normalize(roiImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		# roiImg = cv2.equalizeHist(roiImg)
		cv2.imshow('pupil',roiImg)

		xPupilAxis, yPupilAxis = getHistograms(roiImg, offset, threshold)

		i = xPupilAxis.index(max(xPupilAxis))
		if (i< len(xPupilAxis)):
			while xPupilAxis[i]>0 and i< len(xPupilAxis):
				i = i+1
			x2 = i
		else:
			isPupilCatched = False

		i = xPupilAxis.index(max(xPupilAxis))
		if (i< len(xPupilAxis)):
			while xPupilAxis[i]>0 and i>0:
				i = i-1
			x1 = i
		else:
			isPupilCatched = False

		i = yPupilAxis.index(max(yPupilAxis))
		if (i< len(yPupilAxis)):
			while yPupilAxis[i]>0 and i< len(yPupilAxis):
				i = i+1
			y2 = i
		else:
			isPupilCatched = False

		i = yPupilAxis.index(max(yPupilAxis))
		if (i< len(yPupilAxis)):
			while yPupilAxis[i]>0 and i>0:
				i = i-1
			y1 = i
		else:
			isPupilCatched = False

		if isPupilCatched:
			pupilDiameterX = x2 - x1
			pupilDiameterY = y2 - y1

			#Calculate the ray of the pupil
			if pupilDiameterX > 0 and pupilDiameterY > 0:
				pupilRadiusX = pupilDiameterX/2
				pupilRadiusY = pupilDiameterY/2
				pupilRadius = (pupilRadiusX+pupilDiameterY)/2
				# cv2.circle(img_copy, (centerX, centerY), pupilRadius, (255,0,0), 2)
				# cv2.imshow('eyePupil', img_copy)
				return pupilRadius
			else:
				return 0
		else:
			return 0


def getPupil(img, centerX, centerY, radius):
	img_copy = np.copy(img)
	#Offset: distance from the center, from where we want to calculat the histogram
	offset = int(float(radius))-20
	threshold = 16

	isPupilCatched = True

	#Roi of pixels around the pupil
	roiY1 = int(centerY)-offset
	roiY2 = int(centerY)+offset
	roiX1 = int(centerX)-offset
	roiX2 = int(centerX)+offset
	if (roiX1>0 and roiX2>0 and roiY1>0 and roiY2>0):
		roiImg = img[roiY1:roiY2,roiX1:roiX2]
		cv2.imshow('pupilNotNorm',roiImg)
		# roiImg = normalize(roiImg)
		roiImg = cv2.normalize(roiImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		# roiImg = cv2.equalizeHist(roiImg)
		cv2.imshow('pupil',roiImg)

		#Array containing the histograms
		xPupilAxis = [0] * (2*offset)
		yPupilAxis = [0] * (2*offset)

		#Calcolo istogramma lungo l asse x e y
		for i in range(len(xPupilAxis)):
			for j in range(len(yPupilAxis)):
				#print (img[centerX+i-offset,centerY+j-offset])
				# roiImg[i,j] = roiImg[i,j]-maxDarkValue
				if roiImg[i,j] < threshold:
					xPupilAxis[j] = xPupilAxis[j]+1
					yPupilAxis[i] = yPupilAxis[i]+1


		i = xPupilAxis.index(max(xPupilAxis))
		if (i< len(xPupilAxis)):
			while xPupilAxis[i]>0 and i< len(xPupilAxis):
				print i
				i = i+1
			x2 = i
		else:
			isPupilCatched = False

		i = xPupilAxis.index(max(xPupilAxis))
		if (i< len(xPupilAxis)):
			while xPupilAxis[i]>0 and i>0:
				print i
				i = i-1
			x1 = i
		else:
			isPupilCatched = False

		i = yPupilAxis.index(max(yPupilAxis))
		if (i< len(yPupilAxis)):
			while yPupilAxis[i]>0 and i< len(yPupilAxis):
				print i
				i = i+1
			y2 = i
		else:
			isPupilCatched = False

		i = yPupilAxis.index(max(yPupilAxis))
		if (i< len(yPupilAxis)):
			while yPupilAxis[i]>0 and i>0:
				print i
				i = i-1
			y1 = i
		else:
			isPupilCatched = False

		if isPupilCatched:
			pupilDiameterX = x2 - x1
			pupilDiameterY = y2 - y1

			#Calculate the ray of the pupil
			if pupilDiameterX > 0 and pupilDiameterY > 0:
				pupilRadiusX = pupilDiameterX/2
				pupilRadiusY = pupilDiameterY/2
				pupilRadius = (pupilRadiusX+pupilDiameterY)/2
				cv2.circle(img_copy, (centerX, centerY), pupilRadius, (255,0,0), 2)
				# cv2.imshow('eyePupil', img_copy)
				return pupilRadius
			else:
				return 0
		else:
			return 0