import argparse
import os
from datetime import datetime
import tensorflow as tf
import random
import numpy as np
import scipy.io
from scipy.ndimage.filters import gaussian_filter1d
import scipy.special
import pickle
from dataPreprocessing import prepareDataCubesForRNN
import sys


class charSeqRNN(object):
    """
    This class encapsulates all the functionality needed for training, loading and running the handwriting decoder RNN.
    To use it, initialize this class and then call .train() or .inference(). It can also be run from the command line (see bottom
    of the script). The args dictionary passed during initialization is used to configure all aspects of its behavior.
    """

    def __init__(self, args):
        """
        Initialize the RNN, datasets, optimizer, and checkpointing in TensorFlow 2
        eager mode. After initialization is complete, we are ready to either train
        (charSeqRNN.train) or infer (charSeqRNN.inference).
        """
        self.args = args
        self._compiled_distributed_step = None

        load_checkpoints = self._list_checkpoint_paths(self.args["loadDir"])
        has_load_ckpt = len(load_checkpoints) > 0

        if self.args["mode"] == "train":
            self.isTraining = True
            if not has_load_ckpt:
                self.loadingInitParams = False
                self.resumeTraining = False
            elif self.args["loadDir"] == self.args["outputDir"]:
                self.loadingInitParams = True
                self.resumeTraining = True
            else:
                self.loadingInitParams = True
                self.resumeTraining = False
        elif self.args["mode"] == "infer":
            self.isTraining = False
            self.loadingInitParams = True
            self.resumeTraining = False
        else:
            raise ValueError("mode must be 'train' or 'infer'.")

        # count how many days of data are specified
        self.nDays = 0
        for t in range(30):
            if "labelsFile_" + str(t) not in self.args.keys():
                self.nDays = t
                break

        # load data, labels, train/test partitions & synthetic .tfrecord files for all days
        (
            neuralCube_all,
            targets_all,
            errWeights_all,
            numBinsPerTrial_all,
            cvIdx_all,
            recordFileSet_all,
        ) = self._loadAllDatasets()

        # define the input & output dimensions of the RNN
        nOutputs = targets_all[0].shape[2]
        nInputs = neuralCube_all[0].shape[2]
        self.nInputs = nInputs
        self.nOutputs = nOutputs

        # this is used later in inference mode
        self.nTrialsInFirstDataset = neuralCube_all[0].shape[0]

        # random variable seeding
        if self.args["seed"] == -1:
            self.args["seed"] = datetime.now().microsecond
        np.random.seed(self.args["seed"])
        tf.random.set_seed(self.args["seed"])

        self.strategy = self._init_strategy()
        self.use_strategy = self.strategy.num_replicas_in_sync > 1
        if self.use_strategy:
            print(
                "Using MirroredStrategy with "
                + str(self.strategy.num_replicas_in_sync)
                + " replicas."
            )

        self.dayToLayerMap = eval(self.args["dayToLayerMap"])
        self.dayToLayerMapTensor = tf.constant(self.dayToLayerMap, dtype=tf.int32)
        self.dayProbability = np.array(eval(self.args["dayProbability"]))
        self.nInpLayers = len(np.unique(self.dayToLayerMap))
        self.is_bidirectional = self.args["directionality"] == "bidirectional"
        self.skipLen = int(self.args["skipLen"])

        with self.strategy.scope():
            self._build_layers_and_variables(nInputs, nOutputs)

        # --------------Dataset pipeline--------------
        self._setup_datasets(
            neuralCube_all,
            targets_all,
            errWeights_all,
            numBinsPerTrial_all,
            cvIdx_all,
            recordFileSet_all,
        )

        os.makedirs(self.args["outputDir"], exist_ok=True)
        with self.strategy.scope():
            self._create_checkpoint_objects()

            # Initialize all variables in the model, potentially loading them if self.loadingInitParams==True
            self._loadAndInitializeVariables()

    def _init_strategy(self):
        """
        Create a distribution strategy. If multiple GPUs are visible, use MirroredStrategy to
        replicate the model across them; otherwise fall back to the default strategy.
        """
        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:  # pragma: no cover - defensive logging
                print("Could not enable memory growth: {}".format(exc))

        logical_gpus = tf.config.list_logical_devices("GPU")
        if len(logical_gpus) > 1:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    def _build_layers_and_variables(self, nInputs, nOutputs):
        biDir = 2 if self.is_bidirectional else 1

        self.rnnStartState = tf.Variable(
            tf.zeros([biDir, 1, self.args["nUnits"]], dtype=tf.float32),
            name="RNN_layer0/startState",
            trainable=bool(self.args["trainableBackEnd"]),
        )

        self.inputFactors_W_all = []
        self.inputFactors_b_all = []
        for inpLayerIdx in range(self.nInpLayers):
            self.inputFactors_W_all.append(
                tf.Variable(
                    np.identity(nInputs).astype(np.float32),
                    name="inputFactors_W_" + str(inpLayerIdx),
                    trainable=bool(self.args["trainableInput"]),
                )
            )
            self.inputFactors_b_all.append(
                tf.Variable(
                    np.zeros([nInputs]).astype(np.float32),
                    name="inputFactors_b_" + str(inpLayerIdx),
                    trainable=bool(self.args["trainableInput"]),
                )
            )

        self.layer1 = self._build_gru_layer("layer1")
        self.layer2 = self._build_gru_layer("layer2")

        # Build layers so weights exist for checkpointing and L2 collection.
        dummy_batch = tf.zeros([1, 1, nInputs], dtype=tf.float32)
        self.layer1(dummy_batch, initial_state=self._initial_state(1), training=False)
        dummy_top = tf.zeros([1, 1, self.args["nUnits"] * biDir], dtype=tf.float32)
        self.layer2(dummy_top, initial_state=self._initial_state(1), training=False)

        self.readout_W = tf.Variable(
            tf.random.normal(
                [biDir * self.args["nUnits"], nOutputs],
                stddev=0.05,
                dtype=tf.float32,
            ),
            name="readout_W",
            trainable=bool(self.args["trainableBackEnd"]),
        )
        self.readout_b = tf.Variable(
            tf.zeros([nOutputs], dtype=tf.float32),
            name="readout_b",
            trainable=bool(self.args["trainableBackEnd"]),
        )

        expIdx = []
        for t in range(int(self.args["timeSteps"] / self.skipLen)):
            expIdx.append(np.zeros([self.skipLen]) + t)
        expIdx = np.concatenate(expIdx).astype(int)
        self.expIdx = tf.constant(expIdx, dtype=tf.int32)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-01,
        )
        self.learning_rate = self.optimizer.learning_rate
        self.global_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="global_step"
        )

    def _build_gru_layer(self, name):
        common_args = dict(
            units=self.args["nUnits"],
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            return_state=True,
            reset_after=True,
            trainable=bool(self.args["trainableBackEnd"]),
        )

        if self.args["directionality"] == "forward":
            return tf.keras.layers.GRU(name=name, **common_args)
        elif self.args["directionality"] == "backward":
            return tf.keras.layers.GRU(name=name, go_backwards=True, **common_args)
        elif self.args["directionality"] == "bidirectional":
            return tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(name=f"{name}_gru", **common_args),
                name=name,
            )
        else:
            raise ValueError(
                f"Unsupported directionality {self.args['directionality']}."
            )

    def _initial_state(self, batch_size_tensor):
        batch_size = tf.cast(batch_size_tensor, tf.int32)
        tiled_state = tf.tile(self.rnnStartState, [1, batch_size, 1])
        if self.is_bidirectional:
            forward_state = tiled_state[0]
            backward_state = tiled_state[1]
            return [forward_state, backward_state]
        return tiled_state[0]

    def _input_layer_for_day(self, dayNum):
        if tf.is_tensor(dayNum):
            layer_idx = tf.gather(self.dayToLayerMapTensor, dayNum)
            stacked_W = tf.stack(self.inputFactors_W_all, axis=0)
            stacked_b = tf.stack(self.inputFactors_b_all, axis=0)
            return tf.gather(stacked_W, layer_idx), tf.gather(stacked_b, layer_idx)

        layer_idx = self.dayToLayerMap[dayNum]
        return self.inputFactors_W_all[layer_idx], self.inputFactors_b_all[layer_idx]

    def _run_layer(self, layer, inputs, initial_state):
        outputs = layer(inputs, initial_state=initial_state, training=self.isTraining)
        if isinstance(layer, tf.keras.layers.Bidirectional):
            return outputs[0]
        return outputs[0]

    def _layer_weight_vars(self, layer):
        if isinstance(layer, tf.keras.layers.Bidirectional):
            return [
                layer.forward_layer.cell.kernel,
                layer.forward_layer.cell.recurrent_kernel,
                layer.backward_layer.cell.kernel,
                layer.backward_layer.cell.recurrent_kernel,
            ]
        return [layer.cell.kernel, layer.cell.recurrent_kernel]

    def _collect_l2_weights(self):
        weight_vars = [self.readout_W]
        weight_vars.extend(self.inputFactors_W_all)
        weight_vars.extend(self._layer_weight_vars(self.layer1))
        weight_vars.extend(self._layer_weight_vars(self.layer2))
        return [w for w in weight_vars if w is not None]

    def _trainable_variables(self):
        vars_list = []
        for w in self.inputFactors_W_all + self.inputFactors_b_all:
            if w.trainable:
                vars_list.append(w)
        if self.rnnStartState.trainable:
            vars_list.append(self.rnnStartState)
        for layer in [self.layer1, self.layer2]:
            vars_list.extend([v for v in layer.trainable_variables if v.trainable])
        if self.readout_W.trainable:
            vars_list.append(self.readout_W)
        if self.readout_b.trainable:
            vars_list.append(self.readout_b)
        return vars_list

    def _distribute_batch(self, tensor):
        if not self.use_strategy:
            return tensor

        num_replicas = self.strategy.num_replicas_in_sync
        batch_dim = tf.shape(tensor)[0]
        tf.debugging.assert_equal(
            batch_dim % num_replicas,
            0,
            message="Batch size must be divisible by the number of replicas for multi-GPU training.",
        )

        splits = tf.split(tensor, num_replicas)
        split_stack = tf.stack(splits)
        return self.strategy.experimental_distribute_values_from_function(
            lambda ctx: split_stack[ctx.replica_id_in_sync_group]
        )

    def _concat_per_replica(self, distributed_tensor, axis=0):
        if not self.use_strategy:
            return distributed_tensor
        return tf.concat(
            self.strategy.experimental_local_results(distributed_tensor), axis=axis
        )

    def _forward_and_loss(
        self, batch_inputs, batch_targets, batch_weight, day_num, global_batch_size=None
    ):
        inp_W, inp_b = self._input_layer_for_day(day_num)
        tiled_W = tf.tile(tf.expand_dims(inp_W, 0), [tf.shape(batch_inputs)[0], 1, 1])
        inputFactors = tf.matmul(batch_inputs, tiled_W) + inp_b
        inputFeatures = inputFactors
        if self.args["smoothInputs"] == 1:
            inputFeatures = gaussSmooth(
                inputFeatures, kernelSD=4 / self.args["rnnBinSize"]
            )

        init_state = self._initial_state(tf.shape(batch_inputs)[0])
        rnn_output = self._run_layer(self.layer1, inputFeatures, init_state)

        skip_inputs = rnn_output[:, 0 :: self.skipLen, :]
        init_state_top = self._initial_state(tf.shape(batch_inputs)[0])
        rnn_output2 = self._run_layer(self.layer2, skip_inputs, init_state_top)

        tiledReadoutWeights = tf.tile(
            tf.expand_dims(self.readout_W, 0), [tf.shape(batch_inputs)[0], 1, 1]
        )
        logitOutput_downsample = (
            tf.matmul(rnn_output2, tiledReadoutWeights) + self.readout_b
        )
        logitOutput = tf.gather(logitOutput_downsample, self.expIdx, axis=1)

        if self.args["outputDelay"] > 0:
            labels = batch_targets[:, : -self.args["outputDelay"], :]
            logits = logitOutput[:, self.args["outputDelay"] :, :]
            bw = batch_weight[:, : -self.args["outputDelay"]]
        else:
            labels = batch_targets
            logits = logitOutput
            bw = batch_weight

        transOut = logits[:, :, -1]
        transLabel = labels[:, :, -1]

        logits_main = logits[:, :, 0:-1]
        labels_main = labels[:, :, 0:-1]

        ceLoss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_main, logits=logits_main
        )
        step_len = tf.cast(self.args["timeSteps"], tf.float32)

        ce_per_example = tf.reduce_sum(bw * ceLoss, axis=1) / step_len

        sqErrLoss = tf.square(tf.sigmoid(transOut) - transLabel)
        sq_per_example = tf.reduce_sum(sqErrLoss, axis=1) / step_len

        per_example_total = ce_per_example + 5 * sq_per_example

        if global_batch_size is None:
            totalErr = tf.reduce_mean(per_example_total)
        else:
            totalErr = tf.nn.compute_average_loss(
                per_example_total,
                global_batch_size=tf.cast(global_batch_size, tf.int32),
            )

        l2cost = tf.constant(0.0, dtype=tf.float32)
        if self.args["l2scale"] > 0:
            l2_terms = [tf.reduce_sum(tf.square(v)) for v in self._collect_l2_weights()]
            if l2_terms:
                l2cost = tf.add_n(l2_terms)

        totalCost = totalErr + l2cost * self.args["l2scale"]

        return {
            "total_cost": totalCost,
            "total_err": totalErr,
            "logitOutput": logitOutput,
            "rnnOutput": rnn_output,
            "inputFeatures": inputFeatures,
        }

    def _set_learning_rate(self, lr):
        if hasattr(self.optimizer.learning_rate, "assign"):
            self.optimizer.learning_rate.assign(lr)
        else:
            self.optimizer.learning_rate = lr
        self.learning_rate = self.optimizer.learning_rate

    def _setup_datasets(
        self,
        neuralCube_all,
        targets_all,
        errWeights_all,
        numBinsPerTrial_all,
        cvIdx_all,
        recordFileSet_all,
    ):
        self.daysWithValData = []
        self.train_synth_iters = []
        self.train_real_iters = []
        self.val_iters = []
        self.inference_iter = None

        if self.isTraining:
            for dayIdx in range(self.nDays):
                synthIter = None
                if self.args["synthBatchSize"] > 0:
                    synthDataset = self._make_synth_dataset(
                        recordFileSet_all[dayIdx],
                        self.args["timeSteps"],
                        self.nInputs,
                        self.nOutputs,
                    )
                    synthIter = iter(synthDataset)

                realDataSize = self.args["batchSize"] - self.args["synthBatchSize"]
                trainIdx = cvIdx_all[dayIdx]["trainIdx"]
                valIdx = cvIdx_all[dayIdx]["testIdx"]

                if realDataSize > 0:
                    realDataset = self._makeTrainingDatasetFromRealData(
                        neuralCube_all[dayIdx][trainIdx, :, :],
                        targets_all[dayIdx][trainIdx, :, :],
                        errWeights_all[dayIdx][trainIdx, :],
                        numBinsPerTrial_all[dayIdx][trainIdx, np.newaxis],
                        realDataSize,
                        addNoise=True,
                    )
                    realIter = iter(realDataset)
                else:
                    realIter = None

                if len(valIdx) == 0:
                    valIter = realIter
                    valDataExists = False
                else:
                    valDataset = self._makeTrainingDatasetFromRealData(
                        neuralCube_all[dayIdx][valIdx, :, :],
                        targets_all[dayIdx][valIdx, :, :],
                        errWeights_all[dayIdx][valIdx, :],
                        numBinsPerTrial_all[dayIdx][valIdx, np.newaxis],
                        self.args["batchSize"],
                        addNoise=False,
                    )
                    valIter = iter(valDataset)
                    valDataExists = True

                self.train_synth_iters.append(synthIter)
                self.train_real_iters.append(realIter)
                self.val_iters.append(valIter)
                if valDataExists:
                    self.daysWithValData.append(dayIdx)
        else:
            newDataset = tf.data.Dataset.from_tensor_slices(
                (
                    neuralCube_all[0].astype(np.float32),
                    targets_all[0].astype(np.float32),
                    errWeights_all[0].astype(np.float32),
                    numBinsPerTrial_all[0].astype(np.int32),
                )
            )
            newDataset = newDataset.batch(self.args["batchSize"])
            newDataset = newDataset.repeat()
            newDataset = newDataset.prefetch(tf.data.AUTOTUNE)
            self.inference_iter = iter(newDataset)

    def _make_synth_dataset(self, record_files, nSteps, nInputs, nOutputs):
        if len(record_files) == 0:
            return None
        mapFnc = lambda singleExample: parseDataset(
            singleExample,
            nSteps,
            nInputs,
            nOutputs,
            whiteNoiseSD=self.args["whiteNoiseSD"],
            constantOffsetSD=self.args["constantOffsetSD"],
            randomWalkSD=self.args["randomWalkSD"],
        )
        newDataset = tf.data.TFRecordDataset(record_files)
        newDataset = newDataset.map(mapFnc, num_parallel_calls=tf.data.AUTOTUNE)
        newDataset = newDataset.shuffle(4).repeat()
        newDataset = newDataset.batch(self.args["synthBatchSize"], drop_remainder=True)
        newDataset = newDataset.prefetch(tf.data.AUTOTUNE)
        return newDataset

    def _get_batch(self, datasetNum, dayNum):
        if self.isTraining:
            if datasetNum % 2 == 0:
                return self._next_train_batch(dayNum)
            return self._next_val_batch(dayNum)

        batch_inputs, batch_targets, batch_weight, _ = next(self.inference_iter)
        return batch_inputs, batch_targets, batch_weight

    def _next_train_batch(self, dayNum):
        synthIter = self.train_synth_iters[dayNum]
        realIter = self.train_real_iters[dayNum]
        if synthIter is None:
            inp, targ, weight, _ = next(realIter)
        elif realIter is None:
            inp, targ, weight = next(synthIter)
        else:
            inp_r, targ_r, weight_r, _ = next(realIter)
            inp_s, targ_s, weight_s = next(synthIter)
            inp = tf.concat([inp_s, inp_r], axis=0)
            targ = tf.concat([targ_s, targ_r], axis=0)
            weight = tf.concat([weight_s, weight_r], axis=0)
        return inp, targ, weight

    def _next_val_batch(self, dayNum):
        valIter = self.val_iters[dayNum]
        if valIter is None:
            return self._next_train_batch(dayNum)
        inp, targ, weight, _ = next(valIter)
        return inp, targ, weight

    def _list_checkpoint_paths(self, directory):
        if directory in ["None", None] or not os.path.isdir(directory):
            return []
        try:
            state = tf.train.get_checkpoint_state(directory)
            if state and state.all_model_checkpoint_paths:
                return state.all_model_checkpoint_paths
        except Exception:
            pass
        latest = tf.train.latest_checkpoint(directory)
        return [latest] if latest else []

    def _create_checkpoint_objects(self):
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            global_step=self.global_step,
            rnnStartState=self.rnnStartState,
            inputFactors_W_all=self.inputFactors_W_all,
            inputFactors_b_all=self.inputFactors_b_all,
            layer1=self.layer1,
            layer2=self.layer2,
            readout_W=self.readout_W,
            readout_b=self.readout_b,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.args["outputDir"],
            max_to_keep=self.args["nCheckToKeep"],
        )

    def _loadAndInitializeVariables(self):
        """
        Initializes all tensorflow variables, optionally loading their values from a specified file.
        """
        checkpoints = self._list_checkpoint_paths(self.args["loadDir"])
        self.startingBatchNum = 0

        if self.loadingInitParams and len(checkpoints) > 0:
            load_idx = self.args["loadCheckpointIdx"]
            if load_idx >= 0 and load_idx < len(checkpoints):
                checkpoint_path = checkpoints[load_idx]
            else:
                checkpoint_path = checkpoints[-1]

            print("Loading from checkpoint: " + checkpoint_path)
            status = self.checkpoint.restore(checkpoint_path)
            status.expect_partial()

            if self.resumeTraining:
                self.startingBatchNum = int(self.global_step.numpy()) + 1
            else:
                self.startingBatchNum = 0

    def _save_checkpoint(self, step):
        self.global_step.assign(step)
        if hasattr(self, "manager"):
            self.manager.save(checkpoint_number=step)

    def train(self):
        """
        The main training loop, running eagerly with manual gradient updates.
        """
        batchTrainStats = np.zeros([self.args["nBatchesToTrain"], 6])
        batchValStats = np.zeros(
            [int(np.ceil(self.args["nBatchesToTrain"] / self.args["batchesPerVal"])), 4]
        )
        i = self.startingBatchNum

        if self.resumeTraining:
            resumedStats = scipy.io.loadmat(
                self.args["outputDir"] + "/intermediateOutput"
            )
            batchTrainStats = resumedStats["batchTrainStats"]
            batchValStats = resumedStats["batchValStats"]

        self._save_checkpoint(i)

        while i < self.args["nBatchesToTrain"]:
            dtStart = datetime.now()

            lr = self.args["learnRateStart"] * (
                1 - i / float(self.args["nBatchesToTrain"])
            ) + self.args["learnRateEnd"] * (i / float(self.args["nBatchesToTrain"]))

            dayNum = np.argwhere(np.random.multinomial(1, self.dayProbability))[0][0]
            datasetNum = 2 * dayNum

            self.global_step.assign(i)

            runResultsTrain = self._runBatch(
                datasetNum=datasetNum,
                dayNum=dayNum,
                lr=lr,
                computeGradient=True,
                doGradientUpdate=True,
            )

            trainAcc = computeFrameAccuracy(
                runResultsTrain["logitOutput"],
                runResultsTrain["targets"],
                runResultsTrain["batchWeight"],
                self.args["outputDelay"],
            )

            totalSeconds = (datetime.now() - dtStart).total_seconds()
            batchTrainStats[i, :] = [
                i,
                runResultsTrain["err"],
                runResultsTrain["gradNorm"],
                trainAcc,
                totalSeconds,
                dayNum,
            ]

            if i % self.args["batchesPerVal"] == 0:
                valSetIdx = int(i / self.args["batchesPerVal"])
                batchValStats[valSetIdx, 0:4], outputSnapshot = (
                    self._validationDiagnostics(
                        i,
                        self.args["batchesPerVal"],
                        lr,
                        totalSeconds,
                        runResultsTrain,
                        trainAcc,
                    )
                )

                scipy.io.savemat(
                    self.args["outputDir"] + "/outputSnapshot", outputSnapshot
                )

            if (
                i >= (self.startingBatchNum + self.args["batchesPerSave"] - 1)
                and i % self.args["batchesPerSave"] == 0
            ):
                scipy.io.savemat(
                    self.args["outputDir"] + "/intermediateOutput",
                    {
                        "batchTrainStats": batchTrainStats,
                        "batchValStats": batchValStats,
                    },
                )

            if i % self.args["batchesPerModelSave"] == 0:
                print("SAVING MODEL")
                self._save_checkpoint(i)

            i += 1

        scipy.io.savemat(
            self.args["outputDir"] + "/finalOutput",
            {"batchTrainStats": batchTrainStats, "batchValStats": batchValStats},
        )

        print("SAVING FINAL MODEL")
        self._save_checkpoint(i)

    def inference(self):
        """
        Runs the RNN on the entire dataset once and returns the result - used at inference time for performance evaluation.
        """

        self.nBatchesForInference = np.ceil(
            self.nTrialsInFirstDataset / self.args["batchSize"]
        ).astype(int)

        allOutputs = []
        allUnits = []
        allInputFeatures = []

        print("Starting inference.")

        for _ in range(self.nBatchesForInference):
            returnDict = self._runBatch(
                datasetNum=0,
                dayNum=0,
                lr=0,
                computeGradient=False,
                doGradientUpdate=False,
            )

            allOutputs.append(returnDict["logitOutput"])
            allInputFeatures.append(returnDict["inputFeatures"])
            allUnits.append(returnDict["output"])

        print("Done with inference.")

        allOutputs = np.concatenate(allOutputs, axis=0)
        allUnits = np.concatenate(allUnits, axis=0)
        allInputFeatures = np.concatenate(allInputFeatures, axis=0)

        allOutputs = allOutputs[0 : self.nTrialsInFirstDataset, :, :]
        allUnits = allUnits[0 : self.nTrialsInFirstDataset, :, :]
        allInputFeatures = allInputFeatures[0 : self.nTrialsInFirstDataset, :, :]

        retDict = {}
        retDict["outputs"] = allOutputs
        retDict["units"] = allUnits
        retDict["inputFeatures"] = allInputFeatures

        saveDict = {}
        saveDict["outputs"] = allOutputs

        if self.args["inferenceOutputFileName"] != "None":
            scipy.io.savemat(self.args["inferenceOutputFileName"], saveDict)

        return retDict

    def _validationDiagnostics(
        self, i, nBatchesPerVal, lr, totalSeconds, runResultsTrain, trainAcc
    ):
        """
        Runs a single minibatch on the validation data and returns performance statistics and a snapshot of key variables for
        diagnostic purposes. The snapshot file can be loaded and plotted by an outside program for real-time feedback of how
        the training process is going.
        """
        if self.daysWithValData == []:
            dayNum = self.nDays - 1
            datasetNum = dayNum * 2
        else:
            randIdx = np.random.randint(len(self.daysWithValData))
            dayNum = self.daysWithValData[randIdx]
            datasetNum = 1 + dayNum * 2

        runResults = self._runBatch(
            datasetNum=datasetNum,
            dayNum=dayNum,
            lr=lr,
            computeGradient=True,
            doGradientUpdate=False,
        )

        valAcc = computeFrameAccuracy(
            runResults["logitOutput"],
            runResults["targets"],
            runResults["batchWeight"],
            self.args["outputDelay"],
        )

        print(
            "Val Batch: "
            + str(i)
            + "/"
            + str(self.args["nBatchesToTrain"])
            + ", valErr: "
            + str(runResults["err"])
            + ", trainErr: "
            + str(runResultsTrain["err"])
            + ", Val Acc.: "
            + str(valAcc)
            + ", Train Acc.: "
            + str(trainAcc)
            + ", grad: "
            + str(runResults["gradNorm"])
            + ", learnRate: "
            + str(lr)
            + ", time: "
            + str(totalSeconds)
        )

        outputSnapshot = {}
        outputSnapshot["inputs"] = runResults["inputFeatures"][0, :, :]
        outputSnapshot["rnnUnits"] = runResults["output"][0, :, :]
        outputSnapshot["charProbOutput"] = runResults["logitOutput"][0, :, 0:-1]
        outputSnapshot["charStartOutput"] = scipy.special.expit(
            runResults["logitOutput"][0, self.args["outputDelay"] :, -1]
        )
        outputSnapshot["charProbTarget"] = runResults["targets"][0, :, 0:-1]
        outputSnapshot["charStartTarget"] = runResults["targets"][0, :, -1]
        outputSnapshot["errorWeight"] = runResults["batchWeight"][0, :]

        return [i, runResults["err"], runResults["gradNorm"], valAcc], outputSnapshot

    def _runBatch(self, datasetNum, dayNum, lr, computeGradient, doGradientUpdate):
        """
        Executes one minibatch with optional gradient computation and update.
        """
        self._set_learning_rate(lr)
        batch_inputs, batch_targets, batch_weight = self._get_batch(datasetNum, dayNum)

        should_distribute = self.use_strategy
        static_batch = batch_inputs.shape[0]
        if should_distribute and static_batch is not None:
            should_distribute = (static_batch % self.strategy.num_replicas_in_sync) == 0
        elif should_distribute:
            dynamic_batch = int(tf.shape(batch_inputs)[0].numpy())
            should_distribute = (
                dynamic_batch % self.strategy.num_replicas_in_sync
            ) == 0

        if should_distribute:
            return self._runBatch_distributed(
                batch_inputs,
                batch_targets,
                batch_weight,
                dayNum,
                computeGradient,
                doGradientUpdate,
            )

        grad_norm_value = 0.0
        trainable_vars = self._trainable_variables()

        if computeGradient:
            with tf.GradientTape() as tape:
                forward_result = self._forward_and_loss(
                    batch_inputs, batch_targets, batch_weight, dayNum
                )
            grads = tape.gradient(forward_result["total_cost"], trainable_vars)
            safe_grads = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(grads, trainable_vars)
            ]

            if len(safe_grads) > 0:
                grad_norm_value = tf.linalg.global_norm(safe_grads)
                clipped_grads, _ = tf.clip_by_global_norm(safe_grads, 10.0)
                clipped_pairs = list(zip(clipped_grads, trainable_vars))
                if doGradientUpdate:
                    self.optimizer.apply_gradients(clipped_pairs)
        else:
            forward_result = self._forward_and_loss(
                batch_inputs, batch_targets, batch_weight, dayNum
            )

        returnDict = {
            "err": forward_result["total_err"].numpy(),
            "inputFeatures": forward_result["inputFeatures"].numpy(),
            "output": forward_result["rnnOutput"].numpy(),
            "targets": batch_targets.numpy(),
            "logitOutput": forward_result["logitOutput"].numpy(),
            "batchWeight": batch_weight.numpy(),
            "gradNorm": float(
                grad_norm_value.numpy()
                if isinstance(grad_norm_value, tf.Tensor)
                else grad_norm_value
            ),
        }
        return returnDict

    def _runBatch_distributed(
        self,
        batch_inputs,
        batch_targets,
        batch_weight,
        dayNum,
        computeGradient,
        doGradientUpdate,
    ):
        """
        Distributed minibatch execution across multiple GPUs using MirroredStrategy.
        """
        if self._compiled_distributed_step is None:
            self._compiled_distributed_step = tf.function(
                self._distributed_step, experimental_relax_shapes=True
            )

        (
            total_err,
            logitOutput,
            rnnOutput,
            inputFeatures,
            grad_norm_value,
            targets,
            batchWeight,
        ) = self._compiled_distributed_step(
            batch_inputs,
            batch_targets,
            batch_weight,
            tf.convert_to_tensor(dayNum, dtype=tf.int32),
            tf.convert_to_tensor(computeGradient),
            tf.convert_to_tensor(doGradientUpdate),
        )

        return {
            "err": total_err.numpy(),
            "inputFeatures": inputFeatures.numpy(),
            "output": rnnOutput.numpy(),
            "targets": targets.numpy(),
            "logitOutput": logitOutput.numpy(),
            "batchWeight": batchWeight.numpy(),
            "gradNorm": float(grad_norm_value.numpy()),
        }

    def _distributed_step(
        self,
        batch_inputs,
        batch_targets,
        batch_weight,
        day_num,
        compute_gradient,
        do_gradient_update,
    ):
        """
        Graph-compiled training/inference step executed across replicas.
        Wrapping `strategy.run` in `tf.function` avoids eager overhead warnings
        and ensures correct multi-GPU execution.
        """
        global_batch_size = tf.shape(batch_inputs)[0]

        dist_inputs = self._distribute_batch(batch_inputs)
        dist_targets = self._distribute_batch(batch_targets)
        dist_weight = self._distribute_batch(batch_weight)

        day_num = tf.cast(day_num, tf.int32)
        compute_gradient = tf.convert_to_tensor(compute_gradient)
        do_gradient_update = tf.convert_to_tensor(do_gradient_update)

        def step_fn(replica_inputs, replica_targets, replica_weight):
            trainable_vars = self._trainable_variables()

            def run_with_grads():
                with tf.GradientTape() as tape:
                    forward_result = self._forward_and_loss(
                        replica_inputs,
                        replica_targets,
                        replica_weight,
                        day_num,
                        global_batch_size,
                    )

                grads = tape.gradient(forward_result["total_cost"], trainable_vars)
                safe_grads = [
                    g if g is not None else tf.zeros_like(v)
                    for g, v in zip(grads, trainable_vars)
                ]

                grad_norm_local = (
                    tf.linalg.global_norm(safe_grads)
                    if len(safe_grads) > 0
                    else tf.constant(0.0, dtype=tf.float32)
                )

                clipped_grads, _ = tf.clip_by_global_norm(safe_grads, 10.0)
                clipped_pairs = list(zip(clipped_grads, trainable_vars))

                def apply_update():
                    self.optimizer.apply_gradients(clipped_pairs)
                    return tf.constant(0.0, dtype=tf.float32)

                tf.cond(
                    do_gradient_update,
                    apply_update,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

                return forward_result, grad_norm_local

            def forward_only():
                forward_result = self._forward_and_loss(
                    replica_inputs,
                    replica_targets,
                    replica_weight,
                    day_num,
                    global_batch_size,
                )
                return forward_result, tf.constant(0.0, dtype=tf.float32)

            forward_result, grad_norm_local = tf.cond(
                compute_gradient, run_with_grads, forward_only
            )

            return (
                forward_result["total_err"],
                forward_result["logitOutput"],
                forward_result["rnnOutput"],
                forward_result["inputFeatures"],
                grad_norm_local,
                replica_targets,
                replica_weight,
            )

        (
            per_replica_err,
            per_replica_logit,
            per_replica_rnn,
            per_replica_inputs,
            per_replica_grad_norm,
            per_replica_targets,
            per_replica_weight,
        ) = self.strategy.run(step_fn, args=(dist_inputs, dist_targets, dist_weight))

        total_err = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_err, axis=None
        )
        grad_norm_value = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_grad_norm, axis=None
        )

        logitOutput = self._concat_per_replica(per_replica_logit)
        rnnOutput = self._concat_per_replica(per_replica_rnn)
        inputFeatures = self._concat_per_replica(per_replica_inputs)
        targets = self._concat_per_replica(per_replica_targets)
        batchWeight = self._concat_per_replica(per_replica_weight)

        return (
            total_err,
            logitOutput,
            rnnOutput,
            inputFeatures,
            grad_norm_value,
            targets,
            batchWeight,
        )

    def _loadAllDatasets(self):
        """
        Loads the labels & data for each day specified in the training args, and returns the relevant variables as data cubes.
        Also collects the file names of all .tfrecord files needed for including the synthetic data.
        """
        neuralCube_all = []
        targets_all = []
        errWeights_all = []
        numBinsPerTrial_all = []
        cvIdx_all = []
        recordFileSet_all = []

        for dayIdx in range(self.nDays):
            neuralData, targets, errWeights, binsPerTrial, cvIdx = (
                prepareDataCubesForRNN(
                    self.args["sentencesFile_" + str(dayIdx)],
                    self.args["singleLettersFile_" + str(dayIdx)],
                    self.args["labelsFile_" + str(dayIdx)],
                    self.args["cvPartitionFile_" + str(dayIdx)],
                    self.args["sessionName_" + str(dayIdx)],
                    self.args["rnnBinSize"],
                    self.args["timeSteps"],
                    self.isTraining,
                )
            )

            neuralCube_all.append(neuralData)
            targets_all.append(targets)
            errWeights_all.append(errWeights)
            numBinsPerTrial_all.append(binsPerTrial)
            cvIdx_all.append(cvIdx)

            synthDir = self.args["syntheticDatasetDir_" + str(dayIdx)]
            if os.path.isdir(synthDir):
                recordFileSet = [
                    os.path.join(synthDir, file) for file in os.listdir(synthDir)
                ]
            else:
                recordFileSet = []

            if self.args["synthBatchSize"] > 0 and len(recordFileSet) == 0:
                sys.exit(
                    "Error! No synthetic files found in directory "
                    + self.args["syntheticDatasetDir_" + str(dayIdx)]
                    + ", exiting."
                )

            random.shuffle(recordFileSet)
            recordFileSet_all.append(recordFileSet)

        return (
            neuralCube_all,
            targets_all,
            errWeights_all,
            numBinsPerTrial_all,
            cvIdx_all,
            recordFileSet_all,
        )

    def _makeTrainingDatasetFromRealData(
        self, inputs, targets, errWeight, numBinsPerTrial, batchSize, addNoise=True
    ):
        """
        Creates a tf.data.Dataset from real data with optional noise augmentation.
        """
        newDataset = tf.data.Dataset.from_tensor_slices(
            (
                inputs.astype(np.float32),
                targets.astype(np.float32),
                errWeight.astype(np.float32),
                numBinsPerTrial.astype(np.int32),
            )
        )

        newDataset = newDataset.shuffle(batchSize * 4).repeat()

        mapFun = (
            lambda inputs, targets, errWeight, numBinsPerTrial: extractSentenceSnippet(
                inputs,
                targets,
                errWeight,
                numBinsPerTrial,
                self.args["timeSteps"],
                self.args["directionality"],
            )
        )
        newDataset = newDataset.map(mapFun, num_parallel_calls=tf.data.AUTOTUNE)

        if addNoise and (
            self.args["constantOffsetSD"] > 0 or self.args["randomWalkSD"] > 0
        ):
            mapFun = lambda inputs, targets, errWeight, numBinsPerTrial: addMeanNoise(
                inputs,
                targets,
                errWeight,
                numBinsPerTrial,
                self.args["constantOffsetSD"],
                self.args["randomWalkSD"],
                self.args["timeSteps"],
            )
            newDataset = newDataset.map(mapFun, num_parallel_calls=tf.data.AUTOTUNE)

        if addNoise and self.args["whiteNoiseSD"] > 0:
            mapFun = lambda inputs, targets, errWeight, numBinsPerTrial: addWhiteNoise(
                inputs,
                targets,
                errWeight,
                numBinsPerTrial,
                self.args["whiteNoiseSD"],
                self.args["timeSteps"],
            )
            newDataset = newDataset.map(mapFun, num_parallel_calls=tf.data.AUTOTUNE)

        newDataset = newDataset.batch(batchSize, drop_remainder=True)
        newDataset = newDataset.prefetch(tf.data.AUTOTUNE)

        return newDataset


def extractSentenceSnippet(
    inputs, targets, errWeight, numBinsPerTrial, nSteps, directionality
):
    """
    Extracts a random snippet of data from the full sentence to use for the mini-batch.
    """
    randomStart = tf.random.uniform(
        [],
        minval=0,
        maxval=tf.maximum(numBinsPerTrial[0] + (nSteps - 100) - 400, 1),
        dtype=tf.dtypes.int32,
    )

    inputsSnippet = inputs[randomStart : (randomStart + nSteps), :]
    targetsSnippet = targets[randomStart : (randomStart + nSteps), :]

    charStarts = tf.where(targetsSnippet[1:, -1] - targetsSnippet[0:-1, -1] >= 0.1)

    def noLetters():
        ews = tf.zeros(shape=[nSteps])
        return ews

    def atLeastOneLetter():
        firstChar = tf.cast(charStarts[0, 0], dtype=tf.int32)
        lastChar = tf.cast(charStarts[-1, 0], dtype=tf.int32)

        if directionality == "unidirectional":
            # if uni-directional, only need to blank out the first part because it's causal with a delay
            ews = tf.concat(
                [
                    tf.zeros(shape=[firstChar]),
                    errWeight[(randomStart + firstChar) : (randomStart + nSteps)],
                ],
                axis=0,
            )
        else:
            # if bi-directional (acausal), we need to blank out the last incomplete character as well so that only fully complete
            # characters are included
            ews = tf.concat(
                [
                    tf.zeros(shape=[firstChar]),
                    errWeight[(randomStart + firstChar) : (randomStart + lastChar)],
                    tf.zeros(shape=[nSteps - lastChar]),
                ],
                axis=0,
            )

        return ews

    errWeightSnippet = tf.cond(
        tf.equal(tf.shape(charStarts)[0], 0), noLetters, atLeastOneLetter
    )

    return inputsSnippet, targetsSnippet, errWeightSnippet, numBinsPerTrial


def addMeanNoise(
    inputs, targets, errWeight, numBinsPerTrial, constantOffsetSD, randomWalkSD, nSteps
):
    """
    Applies mean drift noise to each time step of the data in the form of constant offsets (sd=sdConstant)
    and random walk noise (sd=sdRandomWalk)
    """
    meanDriftNoise = tf.random.normal(
        [1, int(inputs.shape[1])], mean=0, stddev=constantOffsetSD
    )
    meanDriftNoise += tf.cumsum(
        tf.random.normal([nSteps, int(inputs.shape[1])], mean=0, stddev=randomWalkSD),
        axis=1,
    )

    return inputs + meanDriftNoise, targets, errWeight, numBinsPerTrial


def addWhiteNoise(inputs, targets, errWeight, numBinsPerTrial, whiteNoiseSD, nSteps):
    """
    Applies white noise to each time step of the data (sd=whiteNoiseSD)
    """
    whiteNoise = tf.random.normal(
        [nSteps, int(inputs.shape[1])], mean=0, stddev=whiteNoiseSD
    )

    return inputs + whiteNoise, targets, errWeight, numBinsPerTrial


def parseDataset(
    singleExample,
    nSteps,
    nInputs,
    nClasses,
    whiteNoiseSD=0.0,
    constantOffsetSD=0.0,
    randomWalkSD=0.0,
):
    """
    Parsing function for the .tfrecord file synthetic data. Returns a synthetic snippet with added noise for training.
    """
    features = {
        "inputs": tf.io.FixedLenFeature((nSteps, nInputs), tf.float32),
        "labels": tf.io.FixedLenFeature((nSteps, nClasses), tf.float32),
        "errWeights": tf.io.FixedLenFeature((nSteps), tf.float32),
    }
    parsedFeatures = tf.io.parse_single_example(singleExample, features)

    noise = tf.random.normal([nSteps, nInputs], mean=0.0, stddev=whiteNoiseSD)

    if constantOffsetSD > 0 or randomWalkSD > 0:
        trainNoise_mn = tf.random.normal([1, nInputs], mean=0, stddev=constantOffsetSD)
        trainNoise_mn += tf.cumsum(
            tf.random.normal([nSteps, nInputs], mean=0, stddev=randomWalkSD), axis=1
        )
        noise += trainNoise_mn

    return (
        parsedFeatures["inputs"] + noise,
        parsedFeatures["labels"],
        parsedFeatures["errWeights"],
    )


def _prepare_initial_state(initRNNState, nBatch, nUnits, direction):
    """
    Prepares the initial state tensor(s) for Keras GRU/Bidirectional layers.

    Args:
        initRNNState (tf.Tensor or None): The initial state tensor from the original function.
                                          Expected shape (1, nBatch, nUnits) for a single layer.
        nBatch (int): Batch size.
        nUnits (int): Number of units in the GRU layer.
        direction (str): The direction of the RNN ('forward', 'backward', or 'bidirectional').

    Returns:
        tf.Tensor or list of tf.Tensor or None: Prepared initial state(s) for Keras.
    """
    if initRNNState is None:
        return None

    if len(initRNNState.shape) != 3:
        raise ValueError(
            f"initRNNState has an unexpected shape: {initRNNState.shape}. "
            f"Expected (num_layers, batch, nUnits)."
        )

    if initRNNState.shape[2] not in (nUnits, None):
        raise ValueError(
            f"initRNNState has an unexpected unit dimension: {initRNNState.shape[2]} "
            f"instead of {nUnits}."
        )

    if initRNNState.shape[1] != nBatch:
        initRNNState = tf.tile(initRNNState, [1, nBatch, 1])

    if direction == "bidirectional":
        backward_state = (
            initRNNState[1]
            if initRNNState.shape[0] > 1
            else tf.zeros_like(initRNNState[0])
        )
        return [initRNNState[0], backward_state]

    return initRNNState[0]


def cudnnGraphSingleLayer(
    nUnits, initRNNState, batchInputs, nSteps, nBatch, nInputs, direction
):
    """
    Construct a single GRU layer using TensorFlow Keras layers, leveraging cuDNN if available
    for speed, and implicitly defining the input shape through the layer's first call.
    Also return the trainable weights for L2 regularization.

    Args:
        nUnits (int): Number of units in the GRU layer.
        initRNNState (tf.Tensor or None): Initial state tensor. Expected shape (1, nBatch, nUnits)
                                         for a single layer. If 'bidirectional', this state is used
                                         for the forward pass, and the backward pass is zero-initialized.
                                         Can be None for zero-initialization.
        batchInputs (tf.Tensor): Input tensor of shape (nBatch, nSteps, nInputs).
        nSteps (int): Number of time steps (kept for signature compatibility, inferred from batchInputs).
        nBatch (int): Batch size (kept for signature compatibility, inferred from batchInputs).
        nInputs (int): Number of input features (kept for signature compatibility, inferred from batchInputs).
        direction (str): Direction of the RNN. One of 'forward', 'backward', or 'bidirectional'.

    Returns:
        tuple: A tuple containing:
            - y_cudnn (tf.Tensor): Output tensor of shape (nBatch, nSteps, nUnits) for 'forward'/'backward',
                                   or (nBatch, nSteps, 2 * nUnits) for 'bidirectional'.
            - weights (list): A list of trainable weight tensors (kernel and recurrent_kernel)
                              for L2 regularization.
    """

    # Keras GRU layers expect input (batch, timesteps, features).
    # The `batchInputs` tensor (nBatch, nSteps, nInputs) already matches this format.
    # No transposition is needed.

    # Prepare initial state based on direction and input state tensor.
    # If initRNNState is None, _prepare_initial_state returns None.
    # Otherwise, it returns (nBatch, nUnits) for unidirectional,
    # or [(nBatch, nUnits), (nBatch, nUnits)] for bidirectional.
    prepared_init_state = _prepare_initial_state(
        initRNNState, nBatch, nUnits, direction
    )

    # Bias initializer: tf.constant_initializer(0.0) is equivalent to Zeros()
    bias_initializer = tf.keras.initializers.Zeros()

    if direction == "forward":
        gru_layer = tf.keras.layers.GRU(
            units=nUnits,
            activation="tanh",  # Default activation for GRU
            recurrent_activation="sigmoid",  # Default recurrent activation for GRU
            use_bias=True,
            bias_initializer=bias_initializer,
            return_sequences=True,  # Output for each timestep
            return_state=True,  # Return the final hidden state (though not returned by this function's tuple)
            reset_after=True,  # Matches cuDNN GRU behavior in TF2 for better performance
        )

        # Call the layer. The first call builds the layer based on input shape.
        outputs, final_state = gru_layer(batchInputs, initial_state=prepared_init_state)

        y_cudnn = outputs

        # Collect weights for L2 regularization: kernel (input weights) and recurrent_kernel (recurrent weights)
        trainable_weights = [gru_layer.cell.kernel, gru_layer.cell.recurrent_kernel]

    elif direction == "backward":
        gru_layer = tf.keras.layers.GRU(
            units=nUnits,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            bias_initializer=bias_initializer,
            return_sequences=True,
            return_state=True,
            go_backwards=True,  # This sets the GRU to process sequences in reverse
            reset_after=True,
        )
        outputs, final_state = gru_layer(batchInputs, initial_state=prepared_init_state)

        y_cudnn = outputs
        trainable_weights = [gru_layer.cell.kernel, gru_layer.cell.recurrent_kernel]

    elif direction == "bidirectional":
        # For Bidirectional, initial_state needs [forward_state, backward_state].
        # `prepared_init_state` handles this by providing a list of two tensors or None.

        bidirectional_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=nUnits,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                bias_initializer=bias_initializer,
                return_sequences=True,
                return_state=True,  # Bidirectional returns (output, forward_h, backward_h)
                reset_after=True,
            ),
            # Default merge_mode is 'concat', which means outputs are concatenated along the feature axis.
            # This results in an output shape of (nBatch, nSteps, 2 * nUnits).
        )

        # Call the bidirectional layer.
        # Outputs will be concatenated. States will be (forward_h, backward_h).
        if prepared_init_state is not None:
            outputs, forward_h, backward_h = bidirectional_layer(
                batchInputs, initial_state=prepared_init_state
            )
        else:
            outputs, forward_h, backward_h = bidirectional_layer(batchInputs)

        y_cudnn = outputs

        # Collect weights from both forward and backward layers for regularization
        trainable_weights = [
            bidirectional_layer.forward_layer.kernel,
            bidirectional_layer.forward_layer.recurrent_kernel,
            bidirectional_layer.backward_layer.kernel,
            bidirectional_layer.backward_layer.recurrent_kernel,
        ]

    else:
        raise ValueError(
            f"Unsupported direction: {direction}. Must be 'forward', 'backward', or 'bidirectional'."
        )

    return y_cudnn, trainable_weights


def gaussSmooth(inputs, kernelSD):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    # get gaussian smoothing kernel
    inp = np.zeros([100])
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)

    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # apply the convolution separately for each feature
    convOut = []
    for x in range(inputs.get_shape()[2]):
        convOut.append(
            tf.nn.conv1d(
                input=inputs[:, :, x, tf.newaxis],
                filters=gaussKernel[:, np.newaxis, np.newaxis].astype(np.float32),
                stride=1,
                padding="SAME",
            )
        )

    # gather the separate convolutions together into a 3d tensor again
    smoothedData = tf.concat(convOut, axis=2)

    return smoothedData


def computeFrameAccuracy(rnnOutput, targets, errWeight, outputDelay):
    """
    Computes a frame-by-frame accuracy percentage given the rnnOutput and the targets, while ignoring
    frames that are masked-out by errWeight and accounting for the RNN's outputDelay.
    """
    # Select all columns but the last one (which is the character start signal) and align rnnOutput to targets
    # while taking into account the output delay.
    bestClass = np.argmax(rnnOutput[:, outputDelay:, 0:-1], axis=2)
    indicatedClass = np.argmax(targets[:, 0:-outputDelay, 0:-1], axis=2)
    bw = errWeight[:, 0:-outputDelay]

    # Mean accuracy is computed by summing number of accurate frames and dividing by total number of valid frames (where bw == 1)
    acc = np.sum(
        bw * np.equal(np.squeeze(bestClass), np.squeeze(indicatedClass))
    ) / np.sum(bw)

    return acc


def getDefaultRNNArgs():
    """
    Makes a default 'args' dictionary with all RNN hyperparameters populated with default values.
    """
    args = {}

    # These arguments define each dataset that will be used for training.
    rootDir = "/home/fwillett/handwritingDatasetsForRelease/"
    dataDirs = ["t5.2019.05.08"]
    cvPart = "HeldOutBlocks"

    for x in range(len(dataDirs)):
        args["timeSeriesFile_" + str(x)] = (
            rootDir
            + "Step2_HMMLabels/"
            + cvPart
            + "/"
            + dataDirs[x]
            + "_timeSeriesLabels.mat"
        )
        args["syntheticDatasetDir_" + str(x)] = (
            rootDir
            + "Step3_SyntheticSentences/"
            + cvPart
            + "/"
            + dataDirs[x]
            + "_syntheticSentences/"
        )
        args["cvPartitionFile_" + str(x)] = (
            rootDir + "trainTestPartitions_" + cvPart + ".mat"
        )
        args["sessionName_" + str(x)] = dataDirs[x]

    # Specify which GPU to use (on multi-gpu machines, this prevents tensorflow from taking over all GPUs)
    args["gpuNumber"] = "0,1,2,3,4,5,6,7"

    # mode can either be 'train' or 'inference'
    args["mode"] = "train"

    # where to save the RNN files
    args["outputDir"] = rootDir + "Step4_RNNTraining/" + cvPart

    # We can load the variables from a previous run, either to resume training (if loadDir==outputDir)
    # or otherwise to complete an entirely new training run. 'loadCheckpointIdx' specifies which checkpoint to load (-1 = latest)
    args["loadDir"] = "None"
    args["loadCheckpointIdx"] = -1

    # number of units in each GRU layer
    args["nUnits"] = 512

    # Specifies how many 10 ms time steps to combine a single bin for RNN processing
    args["rnnBinSize"] = 2

    # Applies Gaussian smoothing if equal to 1
    args["smoothInputs"] = 1

    # For the top GRU layer, how many bins to skip for each update (the top layer runs at a slower frequency)
    args["skipLen"] = 5

    # How many bins to delay the output. Some delay is needed in order to give the RNN enough time to see the entire character
    # before deciding on its identity. Default is 1 second (50 bins).
    args["outputDelay"] = 50

    # Can be 'unidrectional' (causal) or 'bidirectional' (acausal)
    args["directionality"] = "forward"

    # standard deivation of the constant-offset firing rate drift noise
    args["constantOffsetSD"] = 0.6

    # standard deviation of the random walk firing rate drift noise
    args["randomWalkSD"] = 0.02

    # standard deivation of the white noise added to the inputs during training
    args["whiteNoiseSD"] = 1.2

    # l2 regularization cost
    args["l2scale"] = 1e-5

    args["learnRateStart"] = 0.01
    args["learnRateEnd"] = 0.0

    # can optionally specify for only the input layers to train or only the back end
    args["trainableInput"] = 1
    args["trainableBackEnd"] = 1

    # this seed is set for numpy and tensorflow when the class is initialized
    args["seed"] = datetime.now().microsecond

    # number of checkpoints to keep saved during training
    args["nCheckToKeep"] = 1

    # how often to save performance statistics
    args["batchesPerSave"] = 200

    # how often to run a validation diagnostic batch
    args["batchesPerVal"] = 50

    # how often to save the model
    args["batchesPerModelSave"] = 5000

    # how many minibatches to use total
    args["nBatchesToTrain"] = 100000

    # number of time steps to use in the minibatch (1200 = 24 seconds)
    args["timeSteps"] = 1200

    # number of sentence snippets to include in the minibatch
    args["batchSize"] = 64

    # how much of each minibatch is synthetic data
    args["synthBatchSize"] = 24

    # can be used to scale up all input features, sometimes useful when transferring to new days without retraining
    args["inputScale"] = 1.0

    # parameters to specify where to save the outputs and which layer to use during inference
    args["inferenceOutputFileName"] = "None"
    args["inferenceInputLayer"] = 0

    # defines the mapping between each day and which input layer to use for that day
    args["dayToLayerMap"] = "[0]"

    # for each day, the probability that a minibatch will pull from that day. Can be used to weight some days more than others
    args["dayProbability"] = "[1.0]"

    return args


# Here we provide support for running from the command line.
# The only command line argument is the name of an args file.
# Launching from the command line is more reliable than launching from within a jupyter notebook, which sometimes hangs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--argsFile", metavar="argsFile", type=str, default="args.p")

    args = parser.parse_args()
    args = vars(args)
    argDict = pickle.load(open(args["argsFile"], "rb"))

    pid = os.getpid()
    parent_pid = os.getppid()
    print(
        "charSeqRnnMigrate starting with PID "
        + str(pid)
        + " (parent PID "
        + str(parent_pid)
        + "). If launched from 04-Train.py, the training process runs in the background "
        + "and will keep running after a Ctrl+C in the launcher. Use this PID to stop it explicitly if needed."
    )

    pid_file = os.path.join(argDict["outputDir"], "charSeqRnn.pid")
    try:
        os.makedirs(argDict["outputDir"], exist_ok=True)
        with open(pid_file, "w") as pf:
            pf.write(str(pid))
        print("Recorded training PID at " + pid_file)
    except Exception as exc:
        print("Warning: could not write PID file (" + str(exc) + ")")

    # set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print("Setting CUDA_VISIBLE_DEVICES to " + argDict["gpuNumber"])
    os.environ["CUDA_VISIBLE_DEVICES"] = argDict["gpuNumber"]

    # instantiate the RNN model
    rnnModel = charSeqRNN(args=argDict)

    # train or infer
    if argDict["mode"] == "train":
        rnnModel.train()
    elif argDict["mode"] == "inference":
        rnnModel.inference()
