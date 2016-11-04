'''
Created on Nov 4, 2016

@author: abhin
'''
import pandas as pd
import numpy as np
import operator
import re

analyticsDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/alalytics.csv")
aIDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/artificial-intelligence.csv")
bigDataDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/bigdata.csv")
dataScienceDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/data-science.csv")
dataVizDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/data-visualization.csv")
mLDf = pd.read_csv("C:/Users/abhin/workspace/Data Science Activity in Stackoverflow/machine-learning.csv")

saveDirPath = "C:/Users/abhin/Documents/DV CourseWork/DV Project/DataScienceRelatedTags/"

analyticsDict = {}
aIDict = {}
bigDataDict = {}
dataScienceDict = {}
dataVizDict = {}
mlDict ={}

def removeSpecialChars(tag):
    tagList = re.findall('<.*?>', tag)
    for i in range(len(tagList)):
        tagList[i] = re.sub('[<>]', '', tagList[i])
    return tagList

def buildTagCount(subjectDict, subjectDF): 
    for tagList in subjectDF.tags:
        for tag in tagList:
            if tag not in subjectDict:
                subjectDict[tag] = 1
            else:
                subjectDict[tag] += 1

def buildRelatedTagDf(tagCountDict):
    relatedTagDf = pd.DataFrame(list(tagCountDict.items()), columns=['tagName', 'tagCount'])
    relatedTagDf = relatedTagDf.sort_values(['tagCount'], ascending=[False])
    return relatedTagDf
                       
analyticsDf.tags = analyticsDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(analyticsDict, analyticsDf)
analyticsRelatedTagsDf = buildRelatedTagDf(analyticsDict)
analyticsRelatedTagsDf.to_csv(saveDirPath+"analyticsTags.csv", index=False)

aIDf.tags = aIDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(aIDict, aIDf)
aIRelatedTagsDf = buildRelatedTagDf(aIDict)
aIRelatedTagsDf.to_csv(saveDirPath+"artificial-IntelligenceTags.csv", index=False)

bigDataDf.tags = bigDataDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(bigDataDict, bigDataDf)
bigDataRelatedTagsDf = buildRelatedTagDf(bigDataDict)
bigDataRelatedTagsDf.to_csv(saveDirPath+"bigDataTags.csv", index=False)

dataScienceDf.tags = dataScienceDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(dataScienceDict, dataScienceDf)
dataScienceRelatedTagsDf = buildRelatedTagDf(dataScienceDict)
dataScienceRelatedTagsDf.to_csv(saveDirPath+"dataScienceTags.csv", index=False)

dataVizDf.tags = dataVizDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(dataVizDict, dataVizDf)
dataVizRelatedTagsDf = buildRelatedTagDf(dataVizDict)
dataVizRelatedTagsDf.to_csv(saveDirPath+"dataVisualizationTags.csv", index=False)

mLDf.tags = mLDf.tags.apply((lambda x: removeSpecialChars(x)))
buildTagCount(mlDict, mLDf)
mLRelatedTagsDf = buildRelatedTagDf(mlDict)
mLRelatedTagsDf.to_csv(saveDirPath+"machineLearningTags.csv", index=False)
