
#This notebook takes a previously trained RNN and evaluates it on held-out data, saving the outputs for later processing.
#We also compute here the overall character error rate and word error rate across all held-out data. 


import tensorflow as tf

#suppress all tensorflow warnings (largely related to compatability with v2)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from datetime import datetime
from charSeqRnnMigrate import charSeqRNN, getDefaultRNNArgs
import argparse

#point this towards the top level dataset directory
rootDir = os.path.expanduser('.') + '/backupBCIData/'

#evaluate the RNN on these datasets
dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']

#use this train/test partition
cvPart = 'HeldOutTrials'

# We will use the default arguments specified here
parser = argparse.ArgumentParser(description='Inference script for RNN.')
parser.add_argument('--gpu', type=str, default='0', help='GPU number to use.')
parser.add_argument('--logdir', type=str, default='', help='Directory for logs.')

parsed_args = parser.parse_args()

#point this towards the specific RNN we want to evaluate
rnnOutputDir = cvPart

#this prevents tensorflow from taking over more than one gpu on a multi-gpu machine
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=parsed_args.gpu

#this is where we're going to save the RNN outputs
inferenceSaveDir = rootDir+'RNNTrainingSteps/Step5_RNNInference/' + rnnOutputDir + '/' + parsed_args.logdir

if not os.path.isdir(rootDir + 'RNNTrainingSteps/Step5_RNNInference'):
    os.mkdir(rootDir + 'RNNTrainingSteps/Step5_RNNInference')
    
if not os.path.isdir(inferenceSaveDir):
    os.mkdir(inferenceSaveDir)


#Configures the RNN for inference mode.
args = getDefaultRNNArgs()

args['outputDir'] = rootDir+'RNNTrainingSteps/Step4_RNNTraining/' + rnnOutputDir + '/' + parsed_args.logdir
args['loadDir'] = args['outputDir']
args['mode'] = 'infer'
args['timeSteps'] = 7500 #Need to specify enough time steps so that the longest sentence fits in the minibatch
args['batchSize'] = 2 #Process just two sentences at a time, to make sure we have enough memory
args['synthBatchSize'] = 0 #turn off synthetic data here, we are only using real data

#Proceeds one dataset at a time. Currently the code is setup to only process a single dataset at inference time,
#so we have to rebuild the graph for each dataset.
for x in range(len(dataDirs)):
    #configure the RNN to process this particular dataset
    print(' ')
    print('Processing dataset ' + dataDirs[x])
    
    args['sentencesFile_0'] = rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat'
    args['singleLettersFile_0'] = rootDir+'Datasets/'+dataDirs[x]+'/singleLetters.mat'
    args['labelsFile_0'] = rootDir+'RNNTrainingSteps/Step2_HMMLabels/'+cvPart+'/'+dataDirs[x]+'_timeSeriesLabels.mat'
    args['syntheticDatasetDir_0'] = rootDir+'RNNTrainingSteps/Step3_SyntheticSentences/'+cvPart+'/'+dataDirs[x]+'_syntheticSentences/'
    args['cvPartitionFile_0'] = rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat'
    args['sessionName_0'] = dataDirs[x]

    args['inferenceOutputFileName'] = inferenceSaveDir + '/' + dataDirs[x] + '_inferenceOutputs.mat'
    args['inferenceInputLayer'] = x
    
    #instantiate the RNN model
    rnnModel = charSeqRNN(args=args)

    #evaluate the RNN on the held-out data
    outputs = rnnModel.inference()
    
    #reset the graph to make space for the next dataset
    tf.compat.v1.reset_default_graph()



#This cell loads the outputs produced above and computes character error counts and word error counts.
from characterDefinitions import getHandwritingCharacterDefinitions
from rnnEval import evaluateRNNOutput, rnnOutputToKaldiMatrices
import warnings

#this stops scipy.io.savemat from throwing a warning about empty entries
warnings.simplefilter(action='ignore', category=FutureWarning)

charDef = getHandwritingCharacterDefinitions()
allErrCounts = []

for x in range(len(dataDirs)):
    print('-- ' + dataDirs[x] + ' --')
    
    #Load up the outputs, which are frame-by-frame probabilities. 
    outputs = scipy.io.loadmat(inferenceSaveDir + '/' + dataDirs[x] + '_inferenceOutputs.mat')
    sentenceDat = scipy.io.loadmat(rootDir+'Datasets/'+dataDirs[x]+'/sentences.mat')
    
    #Convert the outputs into character sequences (with simple thresholding) & get word/character error counts.
    errCounts, decSentences = evaluateRNNOutput(outputs['outputs'], 
                                        sentenceDat['numTimeBinsPerSentence']/2 + 50, 
                                        sentenceDat['sentencePrompt'], 
                                        charDef, 
                                        charStartThresh=0.3, 
                                        charStartDelay=15)
    
    #save decoded sentences, character error rates and word error rates for later summarization
    saveDict = {}
    saveDict['decSentences'] = decSentences
    saveDict['trueSentences'] = sentenceDat['sentencePrompt']
    saveDict.update(errCounts)
    
    scipy.io.savemat(inferenceSaveDir + '/' + dataDirs[x] + '_errCounts.mat', saveDict)
    
    #print results for the validation sentences
    cvPartFile = scipy.io.loadmat(rootDir+'RNNTrainingSteps/trainTestPartitions_'+cvPart+'.mat')
    valIdx = cvPartFile[dataDirs[x]+'_test']
    
    if len(valIdx)==0:
        print('No validation sentences for this session.')
        print('  ')
        continue
            
    valAcc = 100*(1 - np.sum(errCounts['charErrors'][valIdx]) / np.sum(errCounts['charCounts'][valIdx]))

    print('Character error rate for this session: %1.2f%%' % float(100-valAcc))
    print('Below is the decoder output for all validation sentences in this session:')
    print(' ')
    
    for v in np.squeeze(valIdx):
        trueText = sentenceDat['sentencePrompt'][v,0][0]
        trueText = trueText.replace('>',' ')
        trueText = trueText.replace('~','.')
        trueText = trueText.replace('#','')
        
        print('#' + str(v) + ':')
        print('True:    ' + trueText)
        print('Decoded: ' + decSentences[v])
        print(' ')
   
    #put together all the error counts from all sessions so we can compute overall error rates below
    allErrCounts.append(np.stack([errCounts['charCounts'][valIdx],
                             errCounts['charErrors'][valIdx],
                             errCounts['wordCounts'][valIdx],
                             errCounts['wordErrors'][valIdx]],axis=0).T)
        


#Summarize character error rate and word error rate across all sessions.
concatErrCounts = np.squeeze(np.concatenate(allErrCounts, axis=0))
cer = 100*(np.sum(concatErrCounts[:,1]) / np.sum(concatErrCounts[:,0]))
wer = 100*(np.sum(concatErrCounts[:,3]) / np.sum(concatErrCounts[:,2]))

print('Character error rate: %1.2f%%' % float(cer))
print('Word error rate: %1.2f%%' % float(wer))
