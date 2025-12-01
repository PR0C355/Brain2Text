# This notebook trains a single RNN using all ten sessions of data and a specified train/test partition
# (can be 'HeldOutBlocks' or 'HeldOutTrials'). The RNN training process is launched in a separate python kernel.
# This notebook then monitors training progress in real-time by loading data files produced by the RNN trainer
# and plotting them here, so you can watch how it learns over time.


import argparse
import os
from charSeqRnnMigrate import getDefaultRNNArgs

# point this towards the top level dataset directory
rootDir = os.path.expanduser(".") + "/backupBCIData/"

# train an RNN using data from these specified sessions
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

# use this train/test partition
cvPart = "HeldOutTrials"

# name of the directory where this RNN run will be saved
rnnOutputDir = cvPart

# all RNN runs are saved in 'Step4_RNNTraining'
if not os.path.isdir(rootDir + "RNNTrainingSteps/Step4_RNNTraining"):
    os.mkdir(rootDir + "RNNTrainingSteps/Step4_RNNTraining")


# We will use the default arguments specified here
parser = argparse.ArgumentParser(description='Training script for RNN.')
parser.add_argument('--gpu', type=str, default='0', help='GPU number to use.')
parser.add_argument('--logdir', type=str, default='', help='Directory for logs.')

parsed_args = parser.parse_args()
args = getDefaultRNNArgs()
args["gpuNumber"] = parsed_args.gpu

# Configure the arguments for a multi-day RNN (that will have a unique input layer for each day)
for x in range(len(dataDirs)):
    args["sentencesFile_" + str(x)] = (
        rootDir + "Datasets/" + dataDirs[x] + "/sentences.mat"
    )
    args["singleLettersFile_" + str(x)] = (
        rootDir + "Datasets/" + dataDirs[x] + "/singleLetters.mat"
    )
    args["labelsFile_" + str(x)] = (
        rootDir
        + "RNNTrainingSteps/Step2_HMMLabels/"
        + cvPart
        + "/"
        + dataDirs[x]
        + "_timeSeriesLabels.mat"
    )
    args["syntheticDatasetDir_" + str(x)] = (
        rootDir
        + "RNNTrainingSteps/Step3_SyntheticSentences/"
        + cvPart
        + "/"
        + dataDirs[x]
        + "_syntheticSentences/"
    )
    args["cvPartitionFile_" + str(x)] = (
        rootDir + "RNNTrainingSteps/trainTestPartitions_" + cvPart + ".mat"
    )
    args["sessionName_" + str(x)] = dataDirs[x]

args["outputDir"] = rootDir + "RNNTrainingSteps/Step4_RNNTraining/" + rnnOutputDir + "/" + parsed_args.logdir
if not os.path.isdir(args["outputDir"]):
    os.mkdir(args["outputDir"])

# this weights each day equally (0.1 probability for each day) and allocates a unique input layer for each day (0-9)
args["dayProbability"] = "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"
args["dayToLayerMap"] = "[0,1,2,3,4,5,6,7,8,9]"



# The following code snippet will launch an RNN training program in a separate python kernel (so it doesn't launch inside
# the jupyter notebook, which can be unstable).
import time
from IPython import display
from scipy.ndimage.filters import gaussian_filter1d
import mlflow
from charSeqRnnMigrate import charSeqRNN

# set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print("Setting CUDA_VISIBLE_DEVICES to " + args["gpuNumber"])
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpuNumber"]

mlflow.set_tracking_uri("https://mission.tumi.dev/mlflow/")

# instantiate the RNN model
rnnModel = charSeqRNN(args=args)

# train or infer
with mlflow.start_run(log_system_metrics=True):
    # Log parameters - filter to only include serializable values
    params_to_log = {}
    for key, value in args.items():
        # Convert value to string if it's not a simple type
        if isinstance(value, (int, float, str, bool)):
            params_to_log[key] = value
        else:
            params_to_log[key] = str(value)
    
    mlflow.log_params(params_to_log)
    
    if args["mode"] == "train":
        rnnModel.train()
    elif args["mode"] == "inference":
        rnnModel.inference()