import cv
import numpy as np
import math

import naoqi
from naoqi import ALProxy

def classifyImage():

	ipAddress = "127.0.0.1"
	port = 9559
	camProxy = ALProxy("ALVideoDevice", ipAddress, port)
	resolution = 2    # VGA
	colorSpace = 11   # RGB

	videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

	# Get a camera image.
	# image[6] contains the image data passed as an array of ASCII chars.
	naoImage = camProxy.getImageRemote(videoClient)
	camProxy.unsubscribe(videoClient)

	# Get the image size and pixel array.
	imageWidth = naoImage[0]
	imageHeight = naoImage[1]
	array = naoImage[6]

	#originalImage = Image.fromstring("RGB", (imageWidth, imageHeight), array)
	originalImage = cv.CreateImageHeader((imageWidth, imageHeight), cv.IPL_DEPTH_8U, 3)
	cv.SetData(originalImage, array)

	# Convert to grayscale
	grayscaleImage = convertToCvGrayscale(originalImage)
	size = cv.GetSize(grayscaleImage)

	# Filter white spots
	##    thresholdImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	##    cv.SetData(thresholdImage, size[0] * size[1] * "0")
	##    threshold = 150.0
	##    maxValue = 255.0
	##    thresholdType = cv.CV_THRESH_BINARY
	##    cv.Threshold(grayscaleImage, thresholdImage, threshold, maxValue, thresholdType)
	##    cv.SaveImage("treshold_cv.png", thresholdImage)
	thresholdImage = grayscaleImage

	# Gaussian Blur
	cv.Smooth(thresholdImage, thresholdImage, cv.CV_GAUSSIAN, 5, 5)

	# Canny edge detection
	edgeImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(edgeImage, size[0] * size[1] * "0")
	cv.Canny(thresholdImage, edgeImage, 30.00, 90.0)

	# Constrain the search space
	centerImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	maskImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(centerImage, size[0] * size[1] * "0")
	cv.SetData(maskImage, size[0] * size[1] * "0")
	cv.Zero(centerImage)
	cv.Zero(maskImage)
	pt1 = (size[0] / 4, size[1] / 4)
	pt2 = (3 * pt1[0], 3 * pt1[1])
	cv.Rectangle(maskImage, pt1, pt2, cv.CV_RGB(255, 255, 255), cv.CV_FILLED)

	cv.Copy(edgeImage, centerImage, maskImage)

	# Detect circle
	storage = cv.CreateMat(size[0], 1, cv.CV_32FC3)
	returnCode = cv.HoughCircles(centerImage, storage, cv.CV_HOUGH_GRADIENT, 8, size[0]/8, 200, 250, 0, 0)
	circles = np.asarray(storage)

	if len(circles) == 0:
		return -1, -1

	center = ( int(circles[0][0][0]), int(circles[0][0][1]) )
	radius = int(circles[0][0][2])
	#cv.Circle(grayscaleImage, center, radius, cv.CV_RGB(0, 0, 0), 2, 8, 0 )

	# Cut out the circle
	circleImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	maskImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(circleImage, size[0] * size[1] * "0")
	cv.SetData(maskImage, size[0] * size[1] * "0")
	cv.Zero(circleImage)
	cv.Zero(maskImage)
	cv.Circle(maskImage, center, radius, cv.CV_RGB(255, 255, 255), cv.CV_FILLED)

	cv.Copy(grayscaleImage, circleImage, maskImage)

	# Calculate if filled
	average = 0
	errorCount = 0
	goodCount = 0
	for i in range(0, size[0]):
		for j in range(0, size[1]):
			try:
				pixel_value = cv.Get2D(circleImage, i, j)
				average += pixel_value[0]
				goodCount += 1
			except:
				errorCount += 1
	print errorCount

	average = average / goodCount

#	if average > 1:
#		return "no"
#	else:
#		return "yes"

	return average, len(circles)

def convertToCvGrayscale(originalImage):
	size = cv.GetSize(originalImage)

	#cv.SaveImage("original_cv.png", originalImage)

	rgbImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
	cv.SetData(rgbImage, originalImage.tostring())
	cv.CvtColor(originalImage, rgbImage, cv.CV_BGR2RGB)

	#cv.SaveImage("rgb_cv.png", rgbImage)

	grayscaleImage = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(grayscaleImage, size[0] * size[1] * "0")
	cv.CvtColor(rgbImage, grayscaleImage, cv.CV_BGR2GRAY)

	#cv.SaveImage("grayscale_cv.png", grayscaleImage)
					   
	return grayscaleImage
