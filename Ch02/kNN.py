'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):

    # inX = the input vector to classify
    # the hole set of practice examples
    # labels = all the classification class

    # numpy.ndarray.shape : the dimension of array. ( n,m ) if n*m array.
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # subtraction between matrix.
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # Calculating distance ( Euclidian distance )

    # Sorting distances
    sortedDistIndicies = distances.argsort()

    # classCount Dictionary Dictionary = HashMap
    classCount={}
    for i in range(k):

        # voteIlabel should be A or B
        voteIlabel = labels[sortedDistIndicies[i]]
        # Dictionary.get(key,default_value)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# For datingTestSet Analysis
def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}

    # Open txt file of Date Information
    fr = open(filename)

    # read lines
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   

    index = 0

    for line in arrayOLines:

        # String.strip() rips off the specific char from start and end
        line = line.strip()

        # returns array of strings splited by the char \t is Tab
        listFromLine = line.split('\t')

        # Storing parsed words
        returnMat[index,:] = listFromLine[0:3]

        # Convert largeDoses, smallDoses, didntLike to 1,2,3 numbers
        # array[-1] = last element of array
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector

    
# for every columns to have the same effect on prediction
# newValue = ( oldValue - min ) / ( max - min )
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():

    # haRatio = 0.10
    hoRatio = 0.10      #hold out 10%

    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    # normalize
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0] # Vertical size of the matrix
    numTestVecs = int(m*hoRatio) # Test with 50% of the data
    # This should be Changed to int to avoid float errors
    errorCount = 0.0

    for i in range(numTestVecs):

        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)

        # classify0( inX, dataSet, labels, k)
        # inX = the input vector to classify
        # dataSet = the hole set of practice examples
        # labels = all the classification class

        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])

        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']

    # receiving datas of the new Person
    # \ (backslash) means nothing. just showing line changes on books.
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    # load file, normalize, make a testing vector
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])

    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)

    print "You will probably like this person: %s" % resultList[classifierResult - 1]

# Changes the bitmap image file to a 1*1024 vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)

    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        # figure out what number does the file represent
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        # figure out what number does the file represent
        classNumStr = int(fileStr.split('_')[0])
        # making the inX for classify0
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        # classify0( inX, dataSet, labels, k)
        # inX = the input vector to classify
        # dataSet = the hole set of practice examples
        # labels = all the classification class
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
