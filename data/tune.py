"""
Hyperparameter tuning using Optuna.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import optuna
import yaml
import mlflow

from data.train import train_detection, train_segmentation


def objective_detection(
    trial: optuna.Trial,
    base_config: Dict,
    config_path: Path,
    epochs_per_trial: int,
    optimize_metric: str,
    fixed_batch_size: Optional[int],
    fixed_lr: Optional[float],
    fixed_weight_decay: Optional[float],
    tune_lr: bool,
    tune_batch_size: bool,
    tune_weight_decay: bool,
) -> float:
    """Objective function for detection hyperparameter tuning."""
    
    # Suggest hyperparameters
    if tune_lr and fixed_lr is None:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    else:
        lr = fixed_lr or base_config["common"]["lr"]
    
    if tune_batch_size and fixed_batch_size is None:
        batch_size = trial.suggest_int("batch_size", 2, 16)
    else:
        batch_size = fixed_batch_size or base_config["common"]["batch_size"]
    
    if tune_weight_decay and fixed_weight_decay is None:
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    else:
        weight_decay = fixed_weight_decay or base_config["common"]["weight_decay"]
    
    # Update config
    config = base_config.copy()
    config["common"]["epochs"] = epochs_per_trial
    config["common"]["batch_size"] = batch_size
    config["common"]["lr"] = lr
    config["common"]["weight_decay"] = weight_decay
    config["run_name"] = f"tuning_trial_{trial.number}"
    
    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    
    # Run training
    print(f"\n{'='*80}")
    print(f"🔍 Trial {trial.number}: lr={lr:.6f}, batch_size={batch_size}, weight_decay={weight_decay:.6f}")
    print(f"{'='*80}")
    
    try:
        # Start MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log parameters for this trial (same format as train.py)
            mlflow.log_param("task_type", "detection")
            for k, v in config.get("common", {}).items():
                # Skip nested dictionaries (they will be logged separately)
                if isinstance(v, dict):
                    continue
                mlflow.log_param(f"common_{k}", v)
            
            # Log detection parameters
            for k, v in config.get("detection", {}).items():
                mlflow.log_param(f"detection_{k}", v)
            
            train_detection(config, config_path)
            
            # Get metrics from MLflow - need to get the latest metric value
            client = mlflow.tracking.MlflowClient()
            run_id = mlflow.active_run().info.run_id
            run = client.get_run(run_id)
            
            # Get latest metric values (MLflow stores history, get the last one)
            metrics = {}
            for metric_name in ["f1_50_detection", "precision_50_detection", "recall_50_detection", "val_loss_detection"]:
                metric_history = client.get_metric_history(run_id, metric_name)
                if metric_history:
                    metrics[metric_name] = metric_history[-1].value
                else:
                    metrics[metric_name] = 0.0 if "loss" not in metric_name else float('inf')
            
            # Extract metric based on optimize_metric
            if optimize_metric == "f1":
                metric_value = metrics.get("f1_50_detection", 0.0)
            elif optimize_metric == "val_loss":
                metric_value = metrics.get("val_loss_detection", float('inf'))
            else:
                metric_value = metrics.get(optimize_metric, 0.0)
            
            # Log to Optuna
            trial.set_user_attr("f1", metrics.get("f1_50_detection", 0.0))
            trial.set_user_attr("precision", metrics.get("precision_50_detection", 0.0))
            trial.set_user_attr("recall", metrics.get("recall_50_detection", 0.0))
            trial.set_user_attr("val_loss", metrics.get("val_loss_detection", float('inf')))
            
            print(f"✅ Trial {trial.number} completed: {optimize_metric}={metric_value:.4f}")
            
            # For val_loss, we want to minimize, so return as-is (study direction is minimize)
            if optimize_metric == "val_loss":
                return metric_value
            return metric_value
            
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        # Return a bad value
        if optimize_metric == "val_loss":
            return float('inf')
        return 0.0


def objective_segmentation(
    trial: optuna.Trial,
    base_config: Dict,
    config_path: Path,
    epochs_per_trial: int,
    optimize_metric: str,
    fixed_batch_size: Optional[int],
    fixed_lr: Optional[float],
    fixed_weight_decay: Optional[float],
    tune_lr: bool,
    tune_batch_size: bool,
    tune_weight_decay: bool,
) -> float:
    """Objective function for segmentation hyperparameter tuning."""
    
    # Suggest hyperparameters
    if tune_lr and fixed_lr is None:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    else:
        lr = fixed_lr or base_config["common"]["lr"]
    
    if tune_batch_size and fixed_batch_size is None:
        batch_size = trial.suggest_int("batch_size", 2, 16)
    else:
        batch_size = fixed_batch_size or base_config["common"]["batch_size"]
    
    if tune_weight_decay and fixed_weight_decay is None:
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    else:
        weight_decay = fixed_weight_decay or base_config["common"]["weight_decay"]
    
    # Update config
    config = base_config.copy()
    config["common"]["epochs"] = epochs_per_trial
    config["common"]["batch_size"] = batch_size
    config["common"]["lr"] = lr
    config["common"]["weight_decay"] = weight_decay
    config["run_name"] = f"tuning_trial_{trial.number}"
    
    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    
    # Run training
    print(f"\n{'='*80}")
    print(f"🔍 Trial {trial.number}: lr={lr:.6f}, batch_size={batch_size}, weight_decay={weight_decay:.6f}")
    print(f"{'='*80}")
    
    try:
        # Start MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log parameters for this trial (same format as train.py)
            mlflow.log_param("task_type", "segmentation")
            for k, v in config.get("common", {}).items():
                # Skip nested dictionaries (they will be logged separately)
                if isinstance(v, dict):
                    continue
                mlflow.log_param(f"common_{k}", v)
            
            # Log segmentation parameters
            for k, v in config.get("segmentation", {}).items():
                mlflow.log_param(f"segmentation_{k}", v)
            
            train_segmentation(config, config_path)
            
            # Get metrics from MLflow - need to get the latest metric value
            client = mlflow.tracking.MlflowClient()
            run_id = mlflow.active_run().info.run_id
            run = client.get_run(run_id)
            
            # Get latest metric values (MLflow stores history, get the last one)
            metrics = {}
            for metric_name in ["miou_segmentation", "pixel_acc_segmentation", "val_loss_segmentation"]:
                metric_history = client.get_metric_history(run_id, metric_name)
                if metric_history:
                    metrics[metric_name] = metric_history[-1].value
                else:
                    metrics[metric_name] = 0.0 if "loss" not in metric_name else float('inf')
            
            # Extract metric based on optimize_metric
            if optimize_metric == "miou":
                metric_value = metrics.get("miou_segmentation", 0.0)
            elif optimize_metric == "pixel_acc":
                metric_value = metrics.get("pixel_acc_segmentation", 0.0)
            elif optimize_metric == "val_loss":
                metric_value = metrics.get("val_loss_segmentation", float('inf'))
            else:
                metric_value = metrics.get(optimize_metric, 0.0)
            
            # Log to Optuna
            trial.set_user_attr("miou", metrics.get("miou_segmentation", 0.0))
            trial.set_user_attr("pixel_acc", metrics.get("pixel_acc_segmentation", 0.0))
            trial.set_user_attr("val_loss", metrics.get("val_loss_segmentation", float('inf')))
            
            print(f"✅ Trial {trial.number} completed: {optimize_metric}={metric_value:.4f}")
            
            # For val_loss, we want to minimize, so return as-is (study direction is minimize)
            if optimize_metric == "val_loss":
                return metric_value
            return metric_value
            
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        # Return a bad value
        if optimize_metric == "val_loss":
            return float('inf')
        return 0.0


def run_tuning(
    task_type: str,
    model_name: str,
    epochs_per_trial: int,
    n_trials: int,
    timeout: int,
    optimize_metric: str,
    splits_dir: str,
    dataset_root: Optional[str] = None,
    run_name: Optional[str] = None,
    fixed_batch_size: Optional[int] = None,
    fixed_lr: Optional[float] = None,
    fixed_weight_decay: Optional[float] = None,
    tune_lr: bool = True,
    tune_batch_size: bool = True,
    tune_weight_decay: bool = True,
    detection_labels_root: Optional[str] = None,
    segmentation_masks_root: Optional[str] = None,
    random_seed: Optional[int] = 42,  # Set to None for non-deterministic results
):
    """Run hyperparameter tuning using Optuna."""
    
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "configs" / "train_config.yaml"
    
    # Load base config
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    
    # Update config with provided values
    base_config["task_type"] = task_type
    if task_type == "detection":
        base_config.setdefault("detection", {})
        base_config["detection"]["model_name"] = model_name
    else:
        base_config.setdefault("segmentation", {})
        base_config["segmentation"]["model_name"] = model_name
    
    if splits_dir:
        base_config["data"]["splits_dir"] = splits_dir
    if dataset_root:
        base_config["data"]["dataset_root"] = dataset_root
    if detection_labels_root:
        base_config["data"]["detection_labels_root"] = detection_labels_root
    if segmentation_masks_root:
        base_config["data"]["segmentation_masks_root"] = segmentation_masks_root
    
    # Set MLflow experiment (same as train.py)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("autonomous_vehicle")
    
    # Create study
    study_name = f"tuning_{task_type}_{model_name}"
    if run_name:
        study_name = f"tuning_{run_name}"
    
    # Set random seed for reproducibility
    # If random_seed is None, each run will produce different results (non-deterministic)
    # If random_seed is set (e.g., 42), same seed will produce same results (deterministic)
    if random_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        print(f"🔒 Deterministic mode: Random seed = {random_seed}")
    else:
        sampler = optuna.samplers.TPESampler()  # No seed = non-deterministic
        print(f"🎲 Non-deterministic mode: Each run will produce different results")
    
    study = optuna.create_study(
        direction="maximize" if optimize_metric != "val_loss" else "minimize",
        study_name=study_name,
        sampler=sampler,
    )
    
    # Create objective function
    if task_type == "detection":
        objective = lambda trial: objective_detection(
            trial, base_config, config_path, epochs_per_trial, optimize_metric,
            fixed_batch_size, fixed_lr, fixed_weight_decay,
            tune_lr, tune_batch_size, tune_weight_decay
        )
    else:
        objective = lambda trial: objective_segmentation(
            trial, base_config, config_path, epochs_per_trial, optimize_metric,
            fixed_batch_size, fixed_lr, fixed_weight_decay,
            tune_lr, tune_batch_size, tune_weight_decay
        )
    
    # Start MLflow run for tuning
    tuning_run_name = f"tuning_{run_name}" if run_name else f"tuning_{task_type}_{model_name}"
    with mlflow.start_run(run_name=tuning_run_name):
        # Log essential parameters for filtering in experiments tab
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("is_tuning", "true")
        mlflow.log_param("tuning_n_trials", n_trials)
        mlflow.log_param("tuning_timeout", timeout)
        mlflow.log_param("tuning_optimize_metric", optimize_metric)
        mlflow.log_param("tuning_epochs_per_trial", epochs_per_trial)
        
        # Log model name (same format as train.py)
        if task_type == "detection":
            mlflow.log_param("detection_model_name", model_name)
        else:
            mlflow.log_param("segmentation_model_name", model_name)
        
        # Run optimization
        print(f"\n{'='*80}")
        print(f"🔍 HYPERPARAMETER TUNING BAŞLADI")
        print(f"{'='*80}")
        print(f"   Task: {task_type}")
        print(f"   Model: {model_name}")
        print(f"   Trials: {n_trials}")
        print(f"   Timeout: {timeout} seconds")
        print(f"   Optimize metric: {optimize_metric}")
        print(f"   Epochs per trial: {epochs_per_trial}")
        print(f"{'='*80}\n")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        # Log best parameters
        print(f"\n{'='*80}")
        print(f"🏆 EN İYİ HYPERPARAMETER'LAR")
        print(f"{'='*80}")
        print(f"   Best value: {study.best_value:.4f}")
        print(f"   Best params:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
            mlflow.log_param(f"best_{key}", value)
        
        mlflow.log_metric(f"best_{optimize_metric}", study.best_value)
        
        # Save best config
        best_config = base_config.copy()
        best_config["common"]["batch_size"] = study.best_params.get("batch_size", base_config["common"]["batch_size"])
        best_config["common"]["lr"] = study.best_params.get("lr", base_config["common"]["lr"])
        best_config["common"]["weight_decay"] = study.best_params.get("weight_decay", base_config["common"]["weight_decay"])
        if run_name:
            best_config["run_name"] = f"{run_name}_best"
        
        best_config_path = project_root / "configs" / "train_config_best.yaml"
        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_config, f)
        
        mlflow.log_artifact(str(best_config_path))
        print(f"\n   ✅ En iyi config kaydedildi: {best_config_path}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--task_type", type=str, required=True, choices=["detection", "segmentation"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs_per_trial", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=7200)  # seconds
    parser.add_argument("--optimize_metric", type=str, default="f1")
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fixed_batch_size", type=int, default=None)
    parser.add_argument("--fixed_lr", type=float, default=None)
    parser.add_argument("--fixed_weight_decay", type=float, default=None)
    parser.add_argument("--tune_lr", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--tune_batch_size", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--tune_weight_decay", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--detection_labels_root", type=str, default=None)
    parser.add_argument("--segmentation_masks_root", type=str, default=None)
    parser.add_argument("--preview_samples", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility (set to -1 for non-deterministic)")
    
    args = parser.parse_args()
    
    # Convert -1 to None for non-deterministic mode
    random_seed = args.random_seed if args.random_seed != -1 else None
    
    run_tuning(
        task_type=args.task_type,
        model_name=args.model_name,
        epochs_per_trial=args.epochs_per_trial,
        n_trials=args.n_trials,
        timeout=args.timeout,
        optimize_metric=args.optimize_metric,
        splits_dir=args.splits_dir,
        dataset_root=args.dataset_root,
        run_name=args.run_name,
        fixed_batch_size=args.fixed_batch_size,
        fixed_lr=args.fixed_lr,
        fixed_weight_decay=args.fixed_weight_decay,
        tune_lr=args.tune_lr,
        tune_batch_size=args.tune_batch_size,
        tune_weight_decay=args.tune_weight_decay,
        detection_labels_root=args.detection_labels_root,
        segmentation_masks_root=args.segmentation_masks_root,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    main()

