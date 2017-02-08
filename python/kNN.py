from numpy import *
import operator

def createDataSet():
  dataSet = array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
  labels = ['A','A','B','B']
  return dataSet,labels


def kNNClassify(newInput,dataSet,labels,k):
  #print "newInput",newInput,"dataSet:",dataSet
  #print "labels:",labels,"k:",k
  #print ""

  numSamples = dataSet.shape[0]
  #print "numSamples:",numSamples

  diff = tile(newInput,(numSamples,1)) - dataSet
  #print "diff:",diff

  squaredDiff = diff **2
  #print "squaredDiff:",squaredDiff

  squaredDist = sum(squaredDiff,axis = 1)
  #print "squaredDist:",squaredDist

  distance = squaredDist ** 0.5
  #print "distance:",distance

  sortedDistIndices = argsort(distance)
  #print "sortedDistIndices:",sortedDistIndices

  classCount = {}
  for i in xrange(k):
    voteLabel = labels[sortedDistIndices[i]]
    #print "voteLabel:",voteLabel,"count:",classCount.get(voteLabel,0)
    classCount[voteLabel]  = classCount.get(voteLabel,0) + 1
  #print "classCount:",classCount

  maxCount = 0
  for key,value in classCount.items():
    #print "key:",key,"value:",value,"maxCount:",maxCount
    if value > maxCount:
      maxCount = value
      maxIndex = key

  return maxIndex
