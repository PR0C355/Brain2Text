"""
Hyperparameter tuning script for RNN training using Optuna.
This script uses Optuna to intelligently search the hyperparameter space
and integrates with MLflow for experiment tracking.

Usage:
    python 04-tune-optuna.py --n_trials 50 --study_name "rnn_tuning_v1"
"""

import argparse
import os
import sys
import numpy as np
import scipy.io
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from datetime import datetime
from charSeqRnnMigrate import getDefaultRNNArgs, charSeqRNN

# Point this towards the top level dataset directory
rootDir = os.path.expanduser(".") + "/backupBCIData/"

# Train an RNN using data from these specified sessions
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

# Use this train/test partition
cvPart = "HeldOutTrials"


def objective(trial, base_output_dir, gpu_number):
    """
    Optuna objective function that defines the hyperparameter search space
    and runs a single training trial.
    
    Args:
        trial: Optuna trial object
        base_output_dir: Base directory for saving trial outputs
        gpu_number: GPU to use for training
        
    Returns:
        Final validation accuracy (metric to maximize)
    """
    
    # Get default arguments
    args = getDefaultRNNArgs()
    
    # =============================================================================
    # HYPERPARAMETER SEARCH SPACE
    # Define which hyperparameters to tune and their ranges
    # =============================================================================
    
    # Model Architecture
    args["nUnits"] = trial.suggest_categorical("nUnits", [256, 512, 768, 1024])
    args["skipLen"] = trial.suggest_int("skipLen", 3, 7)
    
    # Learning Rate Schedule
    args["learnRateStart"] = trial.suggest_float("learnRateStart", 0.001, 0.05, log=True)
    args["learnRateEnd"] = trial.suggest_float("learnRateEnd", 0.0, 0.001)
    
    # Regularization
    args["l2scale"] = trial.suggest_float("l2scale", 1e-7, 1e-4, log=True)
    
    # Data Augmentation Noise
    args["whiteNoiseSD"] = trial.suggest_float("whiteNoiseSD", 0.4, 2.0)
    args["constantOffsetSD"] = trial.suggest_float("constantOffsetSD", 0.2, 1.2)
    args["randomWalkSD"] = trial.suggest_float("randomWalkSD", 0.005, 0.05)
    
    # Batch Configuration
    args["batchSize"] = trial.suggest_categorical("batchSize", [32, 64, 96, 128])
    # Ensure synthBatchSize is less than batchSize
    max_synth = args["batchSize"] // 2
    args["synthBatchSize"] = trial.suggest_int("synthBatchSize", 8, max_synth)
    
    # Optional: Uncomment to tune these as well
    # args["timeSteps"] = trial.suggest_categorical("timeSteps", [800, 1000, 1200, 1400])
    # args["outputDelay"] = trial.suggest_int("outputDelay", 30, 70)
    # args["rnnBinSize"] = trial.suggest_categorical("rnnBinSize", [1, 2, 3])
    
    # =============================================================================
    # SETUP FOR THIS TRIAL
    # =============================================================================
    
    # Set GPU
    args["gpuNumber"] = str(gpu_number)
    
    # Configure dataset paths
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
    
    # Create unique output directory for this trial
    trial_dir = f"{base_output_dir}/trial_{trial.number:04d}"
    args["outputDir"] = trial_dir
    if not os.path.isdir(args["outputDir"]):
        os.makedirs(args["outputDir"])
    
    # Day configuration
    args["dayProbability"] = "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"
    args["dayToLayerMap"] = "[0,1,2,3,4,5,6,7,8,9]"
    
    # Reduce training iterations for hyperparameter search (optional)
    # You can reduce this to speed up the search, then retrain best model for longer
    # args["nBatchesToTrain"] = 50000  # Reduced for faster tuning
    
    # =============================================================================
    # RUN TRAINING
    # =============================================================================
    
    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpuNumber"]
    
    mlflow.set_tracking_uri("https://mission.tumi.dev/mlflow/")
    
    try:
        # Instantiate and train the RNN
        with mlflow.start_run(nested=True):
            # Log all parameters (convert non-serializable to strings)
            params_to_log = {}
            for key, value in args.items():
                if isinstance(value, (int, float, str, bool)):
                    params_to_log[key] = value
                else:
                    params_to_log[key] = str(value)
            
            mlflow.log_params(params_to_log)
            mlflow.set_tag("trial_number", trial.number)
            
            # Train the model
            rnnModel = charSeqRNN(args=args)
            rnnModel.train()
        
        # =============================================================================
        # EXTRACT FINAL METRICS
        # =============================================================================
        
        # Load final output to get validation metrics
        final_output_path = os.path.join(args["outputDir"], "finalOutput.mat")
        if not os.path.exists(final_output_path):
            print(f"Warning: finalOutput.mat not found for trial {trial.number}")
            return 0.0
        
        finalOutput = scipy.io.loadmat(final_output_path)
        batchValStats = finalOutput["batchValStats"]
        
        # Find the last valid validation entry
        valid_entries = np.where(batchValStats[:, 0] > 0)[0]
        if len(valid_entries) == 0:
            print(f"Warning: No valid validation data for trial {trial.number}")
            return 0.0
        
        last_valid_idx = valid_entries[-1]
        
        # Extract metrics
        final_val_error = float(batchValStats[last_valid_idx, 1])
        final_val_accuracy = float(batchValStats[last_valid_idx, 3])
        
        # Also get training metrics for comparison
        batchTrainStats = finalOutput["batchTrainStats"]
        final_window = max(1, int(0.1 * len(batchTrainStats)))
        final_train_error = float(np.mean(batchTrainStats[-final_window:, 1]))
        final_train_accuracy = float(np.mean(batchTrainStats[-final_window:, 3]))
        
        # Log final metrics
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
            mlflow.log_metrics({
                "final_val_accuracy": final_val_accuracy,
                "final_val_error": final_val_error,
                "final_train_accuracy": final_train_accuracy,
                "final_train_error": final_train_error,
                "overfit_gap": final_train_accuracy - final_val_accuracy
            })
        
        print(f"\nTrial {trial.number} completed:")
        print(f"  Val Accuracy: {final_val_accuracy:.4f}")
        print(f"  Val Error: {final_val_error:.4f}")
        print(f"  Train Accuracy: {final_train_accuracy:.4f}")
        print(f"  Overfit Gap: {final_train_accuracy - final_val_accuracy:.4f}\n")
        
        # Return the metric to optimize (maximize validation accuracy)
        return final_val_accuracy
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log the failure
        with mlflow.start_run(nested=True):
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
        
        # Return a poor score to indicate failure
        return 0.0


def main():
    """Main function to run Optuna hyperparameter optimization."""
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for RNN using Optuna'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of hyperparameter trials to run'
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default=f'rnn_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Name for the Optuna study'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU number to use'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='sqlite:///optuna_studies.db',
        help='Database URL for storing study results'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='tpe',
        choices=['tpe', 'random', 'grid', 'cmaes'],
        help='Optuna sampler to use (tpe=Bayesian, random, grid, cmaes)'
    )
    parser.add_argument(
        '--pruner',
        type=str,
        default='none',
        choices=['none', 'median', 'percentile'],
        help='Pruner for early stopping of unpromising trials'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume an existing study if it exists'
    )
    
    args = parser.parse_args()
    
    # Create base output directory for all trials
    base_output_dir = rootDir + "RNNTrainingSteps/Step4_RNNTraining/" + cvPart + "/optuna_tuning"
    if not os.path.isdir(base_output_dir):
        os.makedirs(base_output_dir)
    
    # =============================================================================
    # CONFIGURE OPTUNA STUDY
    # =============================================================================
    
    # Select sampler
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif args.sampler == 'grid':
        # For grid search, you'd need to define a search space
        sampler = optuna.samplers.GridSampler({})
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    
    # Select pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner()
    elif args.pruner == 'percentile':
        pruner = optuna.pruners.PercentilePruner(25.0)
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        direction="maximize",  # Maximize validation accuracy
        sampler=sampler,
        pruner=pruner
    )
    
    print(f"\n{'='*80}")
    print(f"Starting Optuna Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Sampler: {args.sampler}")
    print(f"Pruner: {args.pruner}")
    print(f"Storage: {args.storage}")
    print(f"GPU: {args.gpu}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}\n")
    
    # Set up MLflow callback for Optuna
    mlflc = MLflowCallback(
        tracking_uri="https://mission.tumi.dev/mlflow/",
        metric_name="val_accuracy",
        create_experiment=True,
        mlflow_kwargs={
            "experiment_name": f"optuna_{args.study_name}",
            "nested": True
        }
    )
    
    # =============================================================================
    # RUN OPTIMIZATION
    # =============================================================================
    
    try:
        study.optimize(
            lambda trial: objective(trial, base_output_dir, args.gpu),
            n_trials=args.n_trials,
            callbacks=[mlflc],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user. Saving current progress...")
    
    # =============================================================================
    # REPORT RESULTS
    # =============================================================================
    
    print(f"\n{'='*80}")
    print(f"Optimization Complete!")
    print(f"{'='*80}\n")
    
    print("Best trial:")
    print(f"  Number: {study.best_trial.number}")
    print(f"  Value (Val Accuracy): {study.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters to file
    best_params_file = os.path.join(base_output_dir, "best_parameters.txt")
    with open(best_params_file, 'w') as f:
        f.write(f"Study: {args.study_name}\n")
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Val Accuracy: {study.best_trial.value:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nBest parameters saved to: {best_params_file}")
    
    # Show importance of hyperparameters (if enough trials completed)
    if len(study.trials) >= 10:
        print("\n" + "="*80)
        print("Hyperparameter Importance:")
        print("="*80)
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                print(f"  {param}: {imp:.4f}")
        except Exception as e:
            print(f"Could not compute importance: {e}")
    
    # Optuna study statistics
    print(f"\n{'='*80}")
    print("Study Statistics:")
    print(f"{'='*80}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    print(f"\n{'='*80}")
    print(f"Results available at: https://mission.tumi.dev/mlflow/")
    print(f"Study database: {args.storage}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
