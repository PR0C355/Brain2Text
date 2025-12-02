
#This notebook uses forced-alginment HMMs to label ALL datasets and BOTH of the 
#train/test partitions ('HeldOutBlocks' and 'HeldOutTrials').

#For a walkthrough of how the data labeling step works see 'Step2_hmmDataLabeling_walkthrough', which steps 
#through a single example sentence while visualizing the key variables.


import numpy as np
import scipy.io
from characterDefinitions import getHandwritingCharacterDefinitions
from dataLabelingStep import labelDataset, constructRNNTargets
import os
import datetime

#point this towards the top level dataset directory
rootDir = os.path.expanduser('.') + '/handwritingBCIData/'

#define which datasets to process
dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']

#defines the list of all 31 characters and what to call them
charDef = getHandwritingCharacterDefinitions()

#saves all labels in this folder
if not os.path.isdir(rootDir + 'RNNTrainingSteps/Step2_HMMLabels'):
    os.mkdir(rootDir + 'RNNTrainingSteps/Step2_HMMLabels')


for dataDir in dataDirs:
    timeStart = datetime.datetime.now()
    print('Labeling ' + dataDir + ' dataset')
    
    #load sentences, single letter, time-warped files, and train/test partitions
    sentenceDat = scipy.io.loadmat(rootDir+'Datasets/'+dataDir+'/sentences.mat')
    singleLetterDat = scipy.io.loadmat(rootDir+'Datasets/'+dataDir+'/singleLetters.mat')
    twCubes = scipy.io.loadmat(rootDir+'RNNTrainingSteps/Step1_TimeWarping/'+dataDir+'_warpedCubes.mat')

    cvPart_heldOutBlocks = scipy.io.loadmat(rootDir+'RNNTrainingSteps/trainTestPartitions_HeldOutBlocks.mat')
    cvPart_heldOutTrials = scipy.io.loadmat(rootDir+'RNNTrainingSteps/trainTestPartitions_HeldOutTrials.mat')
    cvParts = [cvPart_heldOutBlocks, cvPart_heldOutTrials]
    
    #the last two sessions have hashmarks (#) to indicate that T5 should take a brief pause
    #here we remove these from the sentence prompts, otherwise the code below will get confused (because # isn't a character)
    for x in range(sentenceDat['sentencePrompt'].shape[0]):
        sentenceDat['sentencePrompt'][x,0][0] = sentenceDat['sentencePrompt'][x,0][0].replace('#','')
    
    cvFolderNames = ['HeldOutBlocks', 'HeldOutTrials']
    
    sentences = sentenceDat['sentencePrompt'][:,0]
    sentenceLens = sentenceDat['numTimeBinsPerSentence'][:,0]
    
    #construct separate labels for each training partition
    for cvPart, cvFolder in zip(cvParts, cvFolderNames):
        print("Labeling '" + cvFolder + "' partition")
        trainPartitionIdx = cvPart[dataDir+'_train']
        testPartitionIdx = cvPart[dataDir+'_test']
        
        #label the data with an iterative forced alignmnet HMM
        letterStarts, letterDurations, blankWindows = labelDataset(sentenceDat, 
                                                                   singleLetterDat, 
                                                                   twCubes,
                                                                   trainPartitionIdx, 
                                                                   testPartitionIdx, 
                                                                   charDef)
        
        #construct targets for supervised learning
        charStartTarget, charProbTarget, ignoreErrorHere = constructRNNTargets(letterStarts, 
                                                                               letterDurations, 
                                                                               sentenceDat['neuralActivityCube'].shape[1], 
                                                                               sentences, 
                                                                               charDef)
        
        saveDict = {}
        saveDict['letterStarts'] = letterStarts
        saveDict['letterDurations'] = letterDurations
        saveDict['charStartTarget'] = charStartTarget.astype(np.float32)
        saveDict['charProbTarget'] = charProbTarget.astype(np.float32)
        saveDict['ignoreErrorHere'] = ignoreErrorHere.astype(np.float32)
        saveDict['blankWindows'] = blankWindows
        saveDict['timeBinsPerSentence'] = sentenceDat['numTimeBinsPerSentence']
        
        if not os.path.isdir(rootDir + 'RNNTrainingSteps/Step2_HMMLabels/'+cvFolder):
            os.mkdir(rootDir + 'RNNTrainingSteps/Step2_HMMLabels/'+cvFolder)
            
        scipy.io.savemat(rootDir + 'RNNTrainingSteps/Step2_HMMLabels/'+cvFolder+'/'+dataDir+'_timeSeriesLabels.mat', saveDict)
        
    timeEnd = datetime.datetime.now()
    print('Total time taken: ' + str((timeEnd - timeStart).total_seconds()) + ' seconds')
    print(' ')






