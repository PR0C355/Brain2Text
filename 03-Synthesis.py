# This notebook generates synthetic sentences to agument the RNN's training data (for BOTH of the train/test partitions and
# ALL ten sessions). Step 3 utilizes the data labels created during Step 2 to rearrange the data into new sentences.
# The output of Step 3 is a set of .tfrecord files that are mixed together with the real data during RNN training.


import tensorflow as tf

# suppress all tensorflow warnings (largely related to compatability with v2)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import sklearn.decomposition
from characterDefinitions import getHandwritingCharacterDefinitions
from makeSyntheticSentences import (
    generateCharacterSequences,
    extractCharacterSnippets,
    addSingleLetterSnippets,
)
import multiprocessing
import datetime
from dataPreprocessing import normalizeSentenceDataCube

# point this towards the top level dataset directory
rootDir = os.path.expanduser(".") + "/backupBCIData/"

# define which datasets to process
dataDirs = [
    "t5.2019.05.08",
    "t5.2019.11.25",
    "t5.2019.12.09",
    "t5.2019.12.11",
    "t5.2019.12.18",
    "t5.2019.12.20",
    "t5.2020.01.06",
    "t5.2020.01.08",
    "t5.2020.01.13",
    "t5.2020.01.15",
]

# construct synthetic data for both training partitions
cvParts = ["HeldOutBlocks", "HeldOutTrials"]

# defines the list of all 31 characters and what to call them
charDef = getHandwritingCharacterDefinitions()

# saves all synthetic sentences & snippet libraries in this folder
if not os.path.isdir(rootDir + "RNNTrainingSteps/Step3_SyntheticSentences"):
    os.mkdir(rootDir + "RNNTrainingSteps/Step3_SyntheticSentences")


# First, we generate snippet libraries for each dataset by cutting out each letter from each sentence. These can then
# be re-arranged into new sequences.
for dataDir in dataDirs:
    print("Processing " + dataDir)

    for cvPart in cvParts:
        print("--" + cvPart)

        # load datasets and train/test partition
        sentenceDat = scipy.io.loadmat(
            rootDir + "Datasets/" + dataDir + "/sentences.mat"
        )
        singleLetterDat = scipy.io.loadmat(
            rootDir + "Datasets/" + dataDir + "/singleLetters.mat"
        )
        twCubes = scipy.io.loadmat(
            rootDir
            + "RNNTrainingSteps/Step1_TimeWarping/"
            + dataDir
            + "_warpedCubes.mat"
        )

        cvPartFile = scipy.io.loadmat(
            rootDir + "RNNTrainingSteps/trainTestPartitions_" + cvPart + ".mat"
        )
        trainPartitionIdx = cvPartFile[dataDir + "_train"]

        # the last two sessions have hashmarks (#) to indicate that T5 should take a brief pause
        # here we remove these from the sentence prompts, otherwise the code below will get confused (because # isn't a character)
        for x in range(sentenceDat["sentencePrompt"].shape[0]):
            sentenceDat["sentencePrompt"][x, 0][0] = sentenceDat["sentencePrompt"][
                x, 0
            ][0].replace("#", "")

        # normalize the neural activity cube
        neuralCube = normalizeSentenceDataCube(sentenceDat, singleLetterDat)

        # load labels
        labels = scipy.io.loadmat(
            rootDir
            + "RNNTrainingSteps/Step2_HMMLabels/"
            + cvPart
            + "/"
            + dataDir
            + "_timeSeriesLabels.mat"
        )

        # cut out character snippets from the data for augmentation
        snippetDict = extractCharacterSnippets(
            labels["letterStarts"],
            labels["blankWindows"],
            neuralCube,
            sentenceDat["sentencePrompt"][:, 0],
            sentenceDat["numTimeBinsPerSentence"][:, 0],
            trainPartitionIdx,
            charDef,
        )

        # add single letter examples
        snippetDict = addSingleLetterSnippets(
            snippetDict, singleLetterDat, twCubes, charDef
        )

        # save results
        if not os.path.isdir(
            rootDir + "RNNTrainingSteps/Step3_SyntheticSentences/" + cvPart
        ):
            os.mkdir(rootDir + "RNNTrainingSteps/Step3_SyntheticSentences/" + cvPart)
        scipy.io.savemat(
            rootDir
            + "RNNTrainingSteps/Step3_SyntheticSentences/"
            + cvPart
            + "/"
            + dataDir
            + "_snippets.mat",
            snippetDict,
        )


# Now we use the above snippet libraries to make synthetic data for each dataset and train/test partition.

#'nParallelProcesses' specifies how many parallel processes to use when generating synthetic data (to speed things up).
# Decrease if it uses too much memory on your machine. (10 uses ~30 GB of RAM)
nParallelProcesses = 65

for dataDir in dataDirs:
    print("Processing " + dataDir)

    for cvPart in cvParts:
        print("--" + cvPart)

        outputDir = (
            rootDir
            + "RNNTrainingSteps/Step3_SyntheticSentences/"
            + cvPart
            + "/"
            + dataDir
            + "_syntheticSentences"
        )
        bashDir = rootDir + "bashScratch"
        repoDir = os.getcwd()

        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)

        if not os.path.isdir(bashDir):
            os.mkdir(bashDir)

        args = {}
        args["nSentences"] = 256
        args["nSteps"] = 2400
        args["binSize"] = 2
        args["wordListFile"] = (
            repoDir + "/wordList/google-10000-english-usa.txt"
        )  # from https://github.com/first20hours/google-10000-english
        args["rareWordFile"] = repoDir + "/wordList/rareWordIdx.mat"
        args["snippetFile"] = (
            rootDir
            + "RNNTrainingSteps/Step3_SyntheticSentences/"
            + cvPart
            + "/"
            + dataDir
            + "_snippets.mat"
        )
        args["accountForPenState"] = 1
        args["charDef"] = getHandwritingCharacterDefinitions()
        args["seed"] = datetime.datetime.now().microsecond

        argList = []
        for x in range(20):
            newArgs = args.copy()
            newArgs["saveFile"] = outputDir + "/bat_" + str(x) + ".tfrecord"
            newArgs["seed"] += x
            argList.append(newArgs)

        pool = multiprocessing.Pool(nParallelProcesses)
        results = pool.map(generateCharacterSequences, argList)

        pool.close()
        pool.join()


# plot an example synthetic sentence
trlIdx = 22

plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.imshow(np.transpose(results[0][0][trlIdx, :, :]), aspect="auto", clim=[-1, 1])
plt.title("Synthetic Neural Data")
plt.ylabel("Electrode #")
plt.xlabel("Time Bin (20 ms)")

plt.subplot(3, 1, 2)
plt.imshow(np.transpose(results[0][1][trlIdx, :, 0:-1]), aspect="auto")
plt.title("Character Probability Targets")
plt.ylabel("Character #")
plt.xlabel("Time Bin (20 ms)")

plt.subplot(3, 1, 3)
plt.plot(results[0][1][trlIdx, :, -1])
plt.plot(results[0][2][trlIdx, :])
plt.xlim([0, results[0][1].shape[1]])
plt.title("Character Start Target & Error Weights")
plt.legend(["Character Start Signal", "Error Weight"])
plt.xlabel("Time Bin (20 ms)")

plt.tight_layout()
plt.show()
