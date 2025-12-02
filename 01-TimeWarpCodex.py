# This notebook performs Step 1 of the RNN training process: time-warping the single letter data so that it
# can be used to initialize the data-labeling HMM. Running this notebook will (slowly) time-warp all 10 sessions
# and save the results in Step1_TimeWarping folder.

# To run this notebook, you'll need the affinewarp package (https://github.com/ahwillia/affinewarp).

import os
import numpy as np
import scipy.io
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from affinewarp import PiecewiseWarping
from characterDefinitions import getHandwritingCharacterDefinitions

# point this towards the top level dataset directory
rootDir = os.path.expanduser(".") + "/handwritingBCIData/"

# this line limits which GPUs CUDA-aware libraries can see (optional; comment out or edit as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # to use all GPUs

# defines all the sessions that will be time-warped
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

# defines the list of all 31 characters and what to call them
charDef = getHandwritingCharacterDefinitions()

# saves all time-warped data in this folder
if not os.path.isdir(rootDir + "RNNTrainingSteps/Step1_TimeWarping"):
    os.makedirs(rootDir + "RNNTrainingSteps/Step1_TimeWarping")


def compute_warp_matrix(x_knots, y_knots, n_timepoints):
    """
    Builds a dense representation of the warping functions on the original clock-time grid.

    Args:
        x_knots (ndarray): Knot x-locations for each trial (fractional clock time).
        y_knots (ndarray): Knot y-locations for each trial (fractional aligned time).
        n_timepoints (int): Number of time bins in the original data.

    Returns:
        ndarray: Matrix of shape (time, trials) with aligned-time value (in bins) for each clock-time bin.
    """
    time_grid = np.linspace(0.0, 1.0, n_timepoints)
    warp_matrix = np.empty((n_timepoints, x_knots.shape[0]))

    for trial_idx in range(x_knots.shape[0]):
        warped_fraction = np.interp(
            time_grid,
            x_knots[trial_idx],
            y_knots[trial_idx],
            left=y_knots[trial_idx, 0],
            right=y_knots[trial_idx, -1],
        )
        warp_matrix[:, trial_idx] = np.clip(
            warped_fraction * (n_timepoints - 1), 0, n_timepoints - 1
        )

    return warp_matrix


# Time-warp all singleLetters.mat files and save them to the Step1_TimeWarping folder
for dataDir in dataDirs:
    print("Warping dataset: " + dataDir)
    dat = scipy.io.loadmat(rootDir + "Datasets/" + dataDir + "/singleLetters.mat")

    # Because baseline firing rates drift over time, we normalize each electrode's firing rate by subtracting
    # its mean firing rate within each block of data (re-centering it). We also divide by each electrode's standard deviation
    # to normalize the units.
    for char in charDef["charList"]:
        neuralCube = dat["neuralActivityCube_" + char].astype(np.float64)

        # get the trials that belong to this character
        trlIdx = []
        for t in range(dat["characterCues"].shape[0]):
            if dat["characterCues"][t, 0] == char:
                trlIdx.append(t)

        # get the block that each trial belonged to
        blockIdx = dat["blockNumsTimeSeries"][dat["goPeriodOnsetTimeBin"][trlIdx]]
        blockIdx = np.squeeze(blockIdx)

        # subtract block-specific means from each trial
        for b in range(dat["blockList"].shape[0]):
            trialsFromThisBlock = np.squeeze(blockIdx == dat["blockList"][b])
            neuralCube[trialsFromThisBlock, :, :] -= dat["meansPerBlock"][
                np.newaxis, b, :
            ]

        # divide by standard deviation to normalize the units
        neuralCube = neuralCube / dat["stdAcrossAllData"][np.newaxis, :, :]

        # replace the original cube with this newly normalized one
        dat["neuralActivityCube_" + char] = neuralCube

    alignedDat = {}

    # The following warps each character one at a time.
    # (this is slow, and could be sped up significantly by warping multiple characters in parallel)
    for char in charDef["charList"]:
        print("Warping character: " + char)

        # Clears the previous character's graph
        # tf.compat.v1.reset_default_graph()

        # Number of components to visualize principal neural dimensions after alignment.
        n_components = 5

        # Smooths the binned spike counts before time-warping to denoise them (this step is key!)
        smoothed_spikes = scipy.ndimage.filters.gaussian_filter1d(
            dat["neuralActivityCube_" + char], 3.0, axis=1
        )

        # Compute principal neural dimensions for visualization.
        flattened = smoothed_spikes.reshape(-1, smoothed_spikes.shape[2])
        _, _, vh = np.linalg.svd(flattened - flattened.mean(axis=0), full_matrices=False)
        neuron_factors = vh.T[:, : min(n_components, vh.shape[0])]

        # fit time-warping model
        model = PiecewiseWarping(
            n_knots=2,
            warp_reg_scale=0.001,
            smoothness_reg_scale=1.0,
        ).fit(
            smoothed_spikes,
            iterations=50,
            warp_iterations=200,
            verbose=False,
        )

        # use the model object to align data
        estimated_aligned_data = model.transform(dat["neuralActivityCube_" + char])
        smoothed_aligned_data = scipy.ndimage.filters.gaussian_filter1d(
            estimated_aligned_data, 3.0, axis=1
        )

        # build dense warping functions on the original time grid
        warp_matrix = compute_warp_matrix(
            model.x_knots,
            model.y_knots,
            dat["neuralActivityCube_" + char].shape[1],
        )

        # store aligned data and time-warping functions
        alignedDat[char] = estimated_aligned_data
        alignedDat[char + "_T"] = warp_matrix.copy()

        # only make plots for the first session (otherwise the notebook gets too big)
        if dataDir != "t5.2019.05.08":
            continue

        # plot the warping functions to make sure they look reasonable (should be subtle deviations from the identity line)
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        plt.plot(warp_matrix, alpha=1)
        plt.axis("square")
        plt.xlabel("Clock time")
        plt.ylabel("Aligned time")
        plt.xlim(0, warp_matrix.shape[0])
        plt.ylim(0, warp_matrix.shape[0])
        plt.title("Learned warping functions")

        # It's helpful also to visualize how the major dimensions in the data were aligned
        # We chose dimension 2 here, because the top dimension isn't as informative (it's just a large spike at movement onset)
        component_to_plot = min(1, neuron_factors.shape[1] - 1)
        plt.subplot(1, 3, 2)
        for t in range(estimated_aligned_data.shape[0]):
            thisTrialActivity = np.matmul(smoothed_spikes[t, :, :], neuron_factors)
            plt.plot(thisTrialActivity[:, component_to_plot])

        plt.title("Unwarped Trials")
        plt.xlabel("Time Step (10 ms)")
        plt.ylabel("Activity in Top Neural Dimension #2")

        plt.subplot(1, 3, 3)
        for t in range(estimated_aligned_data.shape[0]):
            thisTrialWarpedActivity = np.matmul(
                smoothed_aligned_data[t, :, :], neuron_factors
            )
            plt.plot(thisTrialWarpedActivity[:, component_to_plot])

        plt.title("Warped Trials")
        plt.xlabel("Time Step (10 ms)")
        plt.ylabel("Activity in Top Neural Dimension #2")

        plt.show()

    # save time-warped characters as a .mat file
    fileName = (
        rootDir + "RNNTrainingSteps/Step1_TimeWarping/" + dataDir + "_warpedCubes.mat"
    )
    print("Saving " + fileName)
    scipy.io.savemat(fileName, alignedDat)
