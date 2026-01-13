import datetime
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import streamlit as st


def render_experiments_tab(task_type: str) -> None:
    """Render experiments view and live metrics."""

    def _list_all_artifacts(run_id: str, base_path: str = "") -> List[str]:
        """Recursively list all artifact relative paths for a run."""
        all_items: List[str] = []
        try:
            artifacts = client.list_artifacts(run_id, path=base_path)
        except Exception:
            return all_items

        for item in artifacts:
            item_path = f"{item.path}" if base_path else item.path
            if item.is_dir:
                all_items.extend(_list_all_artifacts(run_id, base_path=item_path))
            else:
                all_items.append(item_path)
        return all_items

    def _render_images(run_id: str, title: str, image_paths: List[str], max_images: int = 12) -> None:
        if not image_paths:
            return
        st.markdown(title)
        cols = st.columns(min(3, len(image_paths)))
        for idx, artifact_path in enumerate(image_paths[:max_images]):
            try:
                local_path = client.download_artifacts(run_id, artifact_path)
                with cols[idx % len(cols)]:
                    st.image(local_path, caption=Path(artifact_path).name, use_column_width=True)
            except Exception:
                # If a single artifact fails, continue rendering others
                continue

    # Display current filter
    task_display = "🎯 Object Detection" if task_type == "detection" else "🔍 Segmentation"
    st.info(f"📋 Showing: **{task_display}** | 💡 Use the sidebar to change task type")

    # Auto-refresh controls
    col_refresh1, col_refresh2 = st.columns([1, 4])
    with col_refresh1:
        auto_refresh = st.checkbox("🔄 Auto refresh", value=False, key="auto_refresh")
    with col_refresh2:
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=5, key="refresh_interval"
            )
        else:
            refresh_interval = None

    if st.button("🔄 Refresh Now", key="manual_refresh"):
        st.rerun()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("autonomous_vehicle")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("autonomous_vehicle")
    if not experiment:
        st.info("No experiments found yet.")
        return
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.task_type = '{task_type}'",
        order_by=["start_time DESC"],  # Newest first
    )
    if not runs:
        st.info(f"No runs for task_type={task_type}")
        return
    # Build dataframe with metrics
    metric_prefix = "detection" if task_type == "detection" else "segmentation"
    metric_train = f"train_loss_{metric_prefix}"
    metric_val = f"val_loss_{metric_prefix}"

    rows = []
    for r in runs:
        run_id = r.info.run_id
        # Get last metric values
        train_hist = client.get_metric_history(run_id, metric_train)
        val_hist = client.get_metric_history(run_id, metric_val)

        last_train_loss = train_hist[-1].value if train_hist else None
        last_val_loss = val_hist[-1].value if val_hist else None

        model_name = r.data.params.get("detection_model_name" if task_type == "detection" else "segmentation_model_name")
        # Truncate model name if too long
        if model_name and len(model_name) > 20:
            model_name = model_name[:17] + "..."

        # Get run name - MLflow stores it in tags or info
        run_name = None
        try:
            if hasattr(r, "data") and hasattr(r.data, "tags"):
                run_name = r.data.tags.get("mlflow.runName") or r.data.tags.get("experiment_name")
            if not run_name and hasattr(r.info, "run_name") and r.info.run_name:
                run_name = r.info.run_name
        except Exception:
            run_name = None

        # Check if this is a tuning run
        is_tuning = r.data.params.get("is_tuning") == "true"
        
        # For tuning runs, show tuning-specific info
        if is_tuning:
            tuning_n_trials = r.data.params.get("tuning_n_trials", "N/A")
            tuning_optimize_metric = r.data.params.get("tuning_optimize_metric", "N/A")
            best_metric_value = None
            # Try to get best metric value
            best_metric_key = f"best_{tuning_optimize_metric}"
            if best_metric_key in r.data.metrics:
                best_metric_value = r.data.metrics[best_metric_key]
            
            rows.append(
                {
                    "run_name": (run_name or "-") + " 🔍",  # Add tuning indicator
                    "run_id": run_id[:8] + "...",  # Shortened for display
                    "start_time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                    "task_type": r.data.params.get("task_type"),
                    "epochs": f"{r.data.params.get('tuning_epochs_per_trial', 'N/A')}/trial",
                    "lr": f"best: {r.data.params.get('best_lr', 'N/A')}",
                    "model_name": model_name or "N/A",
                    "train_loss": f"Trials: {tuning_n_trials}",
                    "val_loss": f"Best {tuning_optimize_metric}: {best_metric_value:.4f}" if best_metric_value is not None else "N/A",
                }
            )
        else:
            # Regular training run
            rows.append(
                {
                    "run_name": run_name or "-",  # Experiment name
                    "run_id": run_id[:8] + "...",  # Shortened for display
                    "start_time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                    "task_type": r.data.params.get("task_type"),
                    "epochs": r.data.params.get("common_epochs"),
                    "lr": r.data.params.get("common_lr"),
                    "model_name": model_name or "N/A",
                    "train_loss": f"{last_train_loss:.4f}" if last_train_loss else "N/A",
                    "val_loss": f"{last_val_loss:.4f}" if last_val_loss else "N/A",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start_time", ascending=False).reset_index(drop=True)

    st.caption(
        f"📅 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 📊 Total {len(runs)} runs"
    )

    df_display = df.copy()
    df_display.insert(0, "Select", False)

    selected_df = st.data_editor(
        df_display,
        width="stretch",
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select to view live metrics or delete",
                default=False,
            ),
        },
        disabled=[
            "run_name",
            "run_id",
            "start_time",
            "task_type",
            "epochs",
            "lr",
            "model_name",
            "train_loss",
            "val_loss",
        ],
    )

    # Delete button for selected runs
    col_delete1, col_delete2, col_delete3 = st.columns([1, 1, 3])
    with col_delete1:
        delete_button = st.button("🗑️ Delete Selected", type="secondary", use_container_width=True)
    with col_delete2:
        if st.session_state.get("show_delete_confirm", False):
            st.warning(f"⚠️ {len(st.session_state.get('runs_to_delete', []))} runs will be deleted!")
    
    if delete_button:
        selected_indices = [i for i in selected_df.index if bool(selected_df.loc[i, "Select"]) is True]
        if selected_indices:
            runs_to_delete = [runs[i].info.run_id for i in selected_indices if i < len(runs)]
            st.session_state["runs_to_delete"] = runs_to_delete
            st.session_state["show_delete_confirm"] = True
            st.rerun()
        else:
            st.warning("⚠️ Please select at least one run to delete!")
    
    # Confirmation dialog
    if st.session_state.get("show_delete_confirm", False):
        st.markdown("---")
        col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 3])
        with col_confirm1:
            if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                runs_to_delete = st.session_state.get("runs_to_delete", [])
                deleted_count = 0
                for run_id in runs_to_delete:
                    try:
                        client.delete_run(run_id)
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"❌ Run deletion error ({run_id[:8]}...): {e}")
                
                st.session_state["show_delete_confirm"] = False
                st.session_state["runs_to_delete"] = []
                st.success(f"✅ {deleted_count} runs deleted successfully!")
                time.sleep(1)
                st.rerun()
        
        with col_confirm2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["show_delete_confirm"] = False
                st.session_state["runs_to_delete"] = []
                st.rerun()

    # Streamlit returns actual booleans; filter manually to avoid pandas truthiness quirks
    selected_indices = [i for i in selected_df.index if bool(selected_df.loc[i, "Select"]) is True]
    table_selected_run_ids = [runs[i].info.run_id for i in selected_indices if i < len(runs)]

    if table_selected_run_ids:
        st.markdown("---")
        st.markdown("### 📊 Live Metrics (Selected Runs)")

        if len(table_selected_run_ids) == 1:
            run_id = table_selected_run_ids[0]
            run_info = next((r for r in runs if r.info.run_id == run_id), None)
            if run_info:
                run_name = (
                    run_info.data.tags.get("mlflow.runName")
                    or run_info.data.tags.get("experiment_name")
                    or run_id[:8]
                )
                st.subheader(f"Run: {run_name}")
                
                # Check if this is a tuning run
                is_tuning = run_info.data.params.get("is_tuning") == "true"
                
                # If tuning run, show nested trial runs
                if is_tuning:
                    st.info("🔍 This is a hyperparameter tuning run. Showing results for all trials below.")
                    
                    # Show best parameters from parent run
                    st.markdown("#### 🏆 Best Hyperparameters")
                    best_params = {
                        "lr": run_info.data.params.get("best_lr", "N/A"),
                        "batch_size": run_info.data.params.get("best_batch_size", "N/A"),
                        "weight_decay": run_info.data.params.get("best_weight_decay", "N/A"),
                        "epochs_per_trial": run_info.data.params.get("tuning_epochs_per_trial", "N/A"),
                    }
                    best_params_df = pd.DataFrame([
                        {"Parameter": k.replace("_", " ").title(), "Value": v}
                        for k, v in best_params.items()
                    ])
                    st.dataframe(best_params_df, hide_index=True, use_container_width=True)
                    
                    # Show best metric value
                    optimize_metric = run_info.data.params.get("tuning_optimize_metric", "f1")
                    best_metric_key = f"best_{optimize_metric}"
                    if best_metric_key in run_info.data.metrics:
                        best_metric_value = run_info.data.metrics[best_metric_key]
                        st.success(f"🏆 Best {optimize_metric.upper()}: {best_metric_value:.4f}")
                    
                    st.markdown("---")
                    
                    # Get nested runs (trials)
                    all_runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
                        order_by=["start_time ASC"],
                    )
                    
                    if all_runs:
                        st.markdown("#### 🔬 Tuning Trials")
                        
                        # Create summary table for all trials
                        trial_rows = []
                        for trial_run in all_runs:
                            trial_num = trial_run.data.tags.get("mlflow.runName", "").replace("trial_", "")
                            
                            # Get trial metrics
                            trial_metric_prefix = "detection" if task_type == "detection" else "segmentation"
                            trial_train_hist = client.get_metric_history(trial_run.info.run_id, f"train_loss_{trial_metric_prefix}")
                            trial_val_hist = client.get_metric_history(trial_run.info.run_id, f"val_loss_{trial_metric_prefix}")
                            
                            # Get best metric value
                            optimize_metric = run_info.data.params.get("tuning_optimize_metric", "f1")
                            if task_type == "detection":
                                if optimize_metric == "f1":
                                    trial_metric_hist = client.get_metric_history(trial_run.info.run_id, "f1_50_detection")
                                elif optimize_metric == "val_loss":
                                    trial_metric_hist = trial_val_hist
                                else:
                                    trial_metric_hist = client.get_metric_history(trial_run.info.run_id, f"{optimize_metric}_50_detection")
                            else:
                                if optimize_metric == "miou":
                                    trial_metric_hist = client.get_metric_history(trial_run.info.run_id, "miou_segmentation")
                                elif optimize_metric == "pixel_acc":
                                    trial_metric_hist = client.get_metric_history(trial_run.info.run_id, "pixel_acc_segmentation")
                                elif optimize_metric == "val_loss":
                                    trial_metric_hist = trial_val_hist
                                else:
                                    trial_metric_hist = client.get_metric_history(trial_run.info.run_id, f"{optimize_metric}_segmentation")
                            
                            best_metric_value = trial_metric_hist[-1].value if trial_metric_hist else None
                            
                            trial_rows.append({
                                "Trial": f"Trial {trial_num}",
                                "lr": trial_run.data.params.get("common_lr", "N/A"),
                                "batch_size": trial_run.data.params.get("common_batch_size", "N/A"),
                                "weight_decay": trial_run.data.params.get("common_weight_decay", "N/A"),
                                "Train Loss": f"{trial_train_hist[-1].value:.4f}" if trial_train_hist else "N/A",
                                "Val Loss": f"{trial_val_hist[-1].value:.4f}" if trial_val_hist else "N/A",
                                f"Best {optimize_metric.upper()}": f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A",
                            })
                        
                        trial_summary_df = pd.DataFrame(trial_rows)
                        st.dataframe(trial_summary_df, hide_index=True, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Show combined loss graphs for all trials
                        st.markdown("#### 📈 Loss Graphs for All Trials")
                        
                        fig_combined, ax_combined = plt.subplots(figsize=(12, 6))
                        colors = plt.cm.tab10(range(len(all_runs)))
                        
                        for idx, trial_run in enumerate(all_runs):
                            trial_run_id = trial_run.info.run_id
                            trial_metric_prefix = "detection" if task_type == "detection" else "segmentation"
                            trial_train_hist = client.get_metric_history(trial_run_id, f"train_loss_{trial_metric_prefix}")
                            trial_val_hist = client.get_metric_history(trial_run_id, f"val_loss_{trial_metric_prefix}")
                            
                            trial_num = trial_run.data.tags.get("mlflow.runName", f"{idx}")
                            
                            if trial_train_hist:
                                trial_train_sorted = sorted(trial_train_hist, key=lambda x: x.step)
                                ax_combined.plot(
                                    [m.step for m in trial_train_sorted],
                                    [m.value for m in trial_train_sorted],
                                    label=f"Trial {trial_num} - Train",
                                    marker="o",
                                    linewidth=1.5,
                                    alpha=0.7,
                                    color=colors[idx],
                                )
                            
                            if trial_val_hist:
                                trial_val_sorted = sorted(trial_val_hist, key=lambda x: x.step)
                                ax_combined.plot(
                                    [m.step for m in trial_val_sorted],
                                    [m.value for m in trial_val_sorted],
                                    label=f"Trial {trial_num} - Val",
                                    marker="s",
                                    linestyle="--",
                                    linewidth=1.5,
                                    alpha=0.7,
                                    color=colors[idx],
                                )
                        
                        ax_combined.set_xlabel("Epoch", fontsize=12)
                        ax_combined.set_ylabel("Loss", fontsize=12)
                        ax_combined.set_title("All Trials Loss Comparison", fontsize=14)
                        ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                        ax_combined.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_combined)
                        
                        # Show best trial's metrics graph (same as normal runs)
                        st.markdown("#### 📊 Best Trial Detailed Metric Charts")
                        
                        # Find best trial
                        optimize_metric = run_info.data.params.get("tuning_optimize_metric", "f1")
                        best_trial = None
                        best_trial_value = float('-inf') if optimize_metric != "val_loss" else float('inf')
                        
                        for trial_run in all_runs:
                            trial_run_id = trial_run.info.run_id
                            trial_metric_prefix = "detection" if task_type == "detection" else "segmentation"
                            
                            if task_type == "detection":
                                if optimize_metric == "f1":
                                    trial_metric_hist = client.get_metric_history(trial_run_id, "f1_50_detection")
                                elif optimize_metric == "val_loss":
                                    trial_val_hist = client.get_metric_history(trial_run_id, f"val_loss_{trial_metric_prefix}")
                                    trial_metric_hist = trial_val_hist
                                else:
                                    trial_metric_hist = client.get_metric_history(trial_run_id, f"{optimize_metric}_50_detection")
                            else:
                                if optimize_metric == "miou":
                                    trial_metric_hist = client.get_metric_history(trial_run_id, "miou_segmentation")
                                elif optimize_metric == "pixel_acc":
                                    trial_metric_hist = client.get_metric_history(trial_run_id, "pixel_acc_segmentation")
                                elif optimize_metric == "val_loss":
                                    trial_val_hist = client.get_metric_history(trial_run_id, f"val_loss_{trial_metric_prefix}")
                                    trial_metric_hist = trial_val_hist
                                else:
                                    trial_metric_hist = client.get_metric_history(trial_run_id, f"{optimize_metric}_segmentation")
                            
                            if trial_metric_hist:
                                trial_best_value = trial_metric_hist[-1].value
                                if optimize_metric == "val_loss":
                                    if trial_best_value < best_trial_value:
                                        best_trial_value = trial_best_value
                                        best_trial = trial_run
                                else:
                                    if trial_best_value > best_trial_value:
                                        best_trial_value = trial_best_value
                                        best_trial = trial_run
                        
                        if best_trial:
                            best_trial_id = best_trial.info.run_id
                            best_trial_num = best_trial.data.tags.get("mlflow.runName", "?")
                            st.info(f"🏆 Best Trial: {best_trial_num} ({optimize_metric.upper()}={best_trial_value:.4f})")
                            
                            # Show best trial's graphs (same format as normal runs)
                            best_trial_metric_prefix = "detection" if task_type == "detection" else "segmentation"
                            best_train_hist = client.get_metric_history(best_trial_id, f"train_loss_{best_trial_metric_prefix}")
                            best_val_hist = client.get_metric_history(best_trial_id, f"val_loss_{best_trial_metric_prefix}")
                            
                            if best_train_hist:
                                best_train_hist = sorted(best_train_hist, key=lambda x: x.step)
                            if best_val_hist:
                                best_val_hist = sorted(best_val_hist, key=lambda x: x.step)
                            
                            if best_train_hist or best_val_hist:
                                # Metrics cards
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    last_train = best_train_hist[-1].value if best_train_hist else None
                                    best_train = min(best_train_hist, key=lambda x: x.value) if best_train_hist else None
                                    best_train_delta = (
                                        f"Best: {best_train.value:.4f} (Epoch {best_train.step})"
                                        if best_train and best_train.value != last_train
                                        else None
                                    )
                                    st.metric("Train Loss", f"{last_train:.4f}" if last_train else "N/A", delta=best_train_delta)
                                
                                with col2:
                                    last_val = best_val_hist[-1].value if best_val_hist else None
                                    best_val = min(best_val_hist, key=lambda x: x.value) if best_val_hist else None
                                    best_val_delta = (
                                        f"Best: {best_val.value:.4f} (Epoch {best_val.step})" if best_val and best_val.value != last_val else None
                                    )
                                    st.metric("Val Loss", f"{last_val:.4f}" if last_val else "N/A", delta=best_val_delta)
                                
                                with col3:
                                    max_epoch = max(best_train_hist[-1].step if best_train_hist else 0, best_val_hist[-1].step if best_val_hist else 0)
                                    total_epochs = best_trial.data.params.get("common_epochs", "?")
                                    st.metric("Epoch", f"{max_epoch}/{total_epochs}")
                                
                                with col4:
                                    status = best_trial.info.status
                                    status_emoji = "🟢" if status == "FINISHED" else "🟡" if status == "RUNNING" else "🔴"
                                    st.metric("Status", f"{status_emoji} {status}")
                                
                                # Loss graph
                                fig_best, ax_best = plt.subplots(figsize=(10, 5))
                                
                                if best_train_hist:
                                    train_sorted = sorted(best_train_hist, key=lambda x: x.step)
                                    train_steps = [m.step for m in train_sorted]
                                    train_values = [m.value for m in train_sorted]
                                    ax_best.plot(train_steps, train_values, label="Train Loss", marker="o", linewidth=2)
                                
                                if best_val_hist:
                                    val_sorted = sorted(best_val_hist, key=lambda x: x.step)
                                    val_steps = [m.step for m in val_sorted]
                                    val_values = [m.value for m in val_sorted]
                                    ax_best.plot(val_steps, val_values, label="Val Loss", marker="s", linestyle="--", linewidth=2)
                                
                                ax_best.set_xlabel("Epoch", fontsize=12)
                                ax_best.set_ylabel("Loss", fontsize=12)
                                ax_best.set_title(f"Best Trial ({best_trial_num}) - Loss Chart", fontsize=14)
                                ax_best.legend()
                                ax_best.grid(True, alpha=0.3)
                                st.pyplot(fig_best)
                                
                                # All metrics graph
                                st.markdown("#### 📊 All Metrics")
                                all_metrics_fig, all_metrics_ax = plt.subplots(figsize=(12, 6))
                                
                                if best_train_hist:
                                    train_sorted = sorted(best_train_hist, key=lambda x: x.step)
                                    all_metrics_ax.plot(
                                        [m.step for m in train_sorted],
                                        [m.value for m in train_sorted],
                                        label="Train Loss",
                                        marker="o",
                                        linewidth=2,
                                        color="#1f77b4",
                                    )
                                
                                if best_val_hist:
                                    val_sorted = sorted(best_val_hist, key=lambda x: x.step)
                                    all_metrics_ax.plot(
                                        [m.step for m in val_sorted],
                                        [m.value for m in val_sorted],
                                        label="Val Loss",
                                        marker="s",
                                        linestyle="--",
                                        linewidth=2,
                                        color="#ff7f0e",
                                    )
                                
                                # Task-specific metrics
                                if task_type == "detection":
                                    for metric_name, display_name, color in [
                                        ("f1_50_detection", "F1 Score", "#2ca02c"),
                                        ("precision_50_detection", "Precision", "#d62728"),
                                        ("recall_50_detection", "Recall", "#9467bd"),
                                    ]:
                                        hist = client.get_metric_history(best_trial_id, metric_name)
                                        if hist:
                                            hist_sorted = sorted(hist, key=lambda x: x.step)
                                            all_metrics_ax.plot(
                                                [m.step for m in hist_sorted],
                                                [m.value for m in hist_sorted],
                                                label=display_name,
                                                marker="^",
                                                linewidth=2,
                                                color=color,
                                            )
                                else:
                                    for metric_name, display_name, color in [
                                        ("pixel_acc_segmentation", "Pixel Accuracy", "#2ca02c"),
                                        ("miou_segmentation", "mIoU", "#d62728"),
                                    ]:
                                        hist = client.get_metric_history(best_trial_id, metric_name)
                                        if hist:
                                            hist_sorted = sorted(hist, key=lambda x: x.step)
                                            all_metrics_ax.plot(
                                                [m.step for m in hist_sorted],
                                                [m.value for m in hist_sorted],
                                                label=display_name,
                                                marker="^",
                                                linewidth=2,
                                                color=color,
                                            )
                                
                                all_metrics_ax.set_xlabel("Epoch", fontsize=12)
                                all_metrics_ax.set_ylabel("Metric Value", fontsize=12)
                                all_metrics_ax.set_title(f"Best Trial ({best_trial_num}) - All Metrics", fontsize=14)
                                all_metrics_ax.legend()
                                all_metrics_ax.grid(True, alpha=0.3)
                                st.pyplot(all_metrics_fig)
                        
                        st.markdown("---")
                        
                        # Show detailed metrics for each trial
                        st.markdown("#### 📊 Detailed Trial Metrics")
                        
                        for idx, trial_run in enumerate(all_runs):
                            with st.expander(f"🔬 Trial {trial_run.data.tags.get('mlflow.runName', f'{idx}')} - Detailed Metrics", expanded=(idx == 0)):
                                trial_run_id = trial_run.info.run_id
                                
                                # Show trial parameters
                                st.markdown(f"**⚙️ Trial Parameters**")
                                
                                # Try multiple parameter name formats
                                trial_params = {}
                                
                                # Try common_* format first (standard format)
                                lr = trial_run.data.params.get("common_lr")
                                if lr is None:
                                    # Try without common_ prefix
                                    lr = trial_run.data.params.get("lr")
                                if lr is None:
                                    # Try from config if available
                                    lr = "N/A"
                                
                                batch_size = trial_run.data.params.get("common_batch_size")
                                if batch_size is None:
                                    batch_size = trial_run.data.params.get("batch_size")
                                if batch_size is None:
                                    batch_size = "N/A"
                                
                                weight_decay = trial_run.data.params.get("common_weight_decay")
                                if weight_decay is None:
                                    weight_decay = trial_run.data.params.get("weight_decay")
                                if weight_decay is None:
                                    weight_decay = "N/A"
                                
                                epochs = trial_run.data.params.get("common_epochs")
                                if epochs is None:
                                    epochs = trial_run.data.params.get("epochs")
                                if epochs is None:
                                    epochs = "N/A"
                                
                                trial_params = {
                                    "lr": lr,
                                    "batch_size": batch_size,
                                    "weight_decay": weight_decay,
                                    "epochs": epochs,
                                }
                                
                                # Debug: Show all available params (temporary, can be removed later)
                                if lr == "N/A" and batch_size == "N/A":
                                    with st.expander("🔍 Debug: All Parameters (for troubleshooting)"):
                                        all_params = list(trial_run.data.params.keys())
                                        st.write(f"A total of {len(all_params)} parameters found:")
                                        for param_key in sorted(all_params):
                                            st.write(f"- `{param_key}`: {trial_run.data.params.get(param_key)}")
                                
                                trial_params_df = pd.DataFrame([
                                    {"Parameter": k.replace("_", " ").title(), "Value": str(v)}
                                    for k, v in trial_params.items()
                                ])
                                st.dataframe(trial_params_df, hide_index=True, use_container_width=True)
                                
                                # Show trial metrics table (same as regular runs)
                                st.markdown("**📊 Metric Summary**")
                                trial_metric_prefix = "detection" if task_type == "detection" else "segmentation"
                                
                                latest_metrics = []
                                
                                if task_type == "detection":
                                    base_metrics = [
                                        ("f1_50_detection", "F1 Score (IoU=0.5)"),
                                        ("precision_50_detection", "Precision (IoU=0.5)"),
                                        ("recall_50_detection", "Recall (IoU=0.5)"),
                                    ]
                                    
                                    for metric_name, display_name in base_metrics:
                                        hist = client.get_metric_history(trial_run_id, metric_name)
                                        if hist:
                                            best_metric = max(hist, key=lambda x: x.value)
                                            last_metric = hist[-1]
                                            latest_metrics.append({
                                                "Metric": display_name,
                                                "Latest Value": f"{last_metric.value:.4f}",
                                                "Latest Epoch": last_metric.step,
                                                "Best Value": f"{best_metric.value:.4f}",
                                                "Best Epoch": best_metric.step,
                                            })
                                else:
                                    base_metrics = [
                                        ("pixel_acc_segmentation", "Pixel Accuracy"),
                                        ("miou_segmentation", "mIoU"),
                                    ]
                                    
                                    class_metric_keys = [
                                        k for k in trial_run.data.metrics.keys() if k.startswith("iou_class_")
                                    ]
                                    for key in sorted(class_metric_keys):
                                        name = key.replace("iou_class_", "").replace("_segmentation", "")
                                        display = f"IoU {name.replace('_', ' ')}"
                                        base_metrics.append((key, display))
                                    
                                    for metric_name, display_name in base_metrics:
                                        hist = client.get_metric_history(trial_run_id, metric_name)
                                        if hist:
                                            best_metric = max(hist, key=lambda x: x.value)
                                            last_metric = hist[-1]
                                            latest_metrics.append({
                                                "Metric": display_name,
                                                "Latest Value": f"{last_metric.value:.4f}",
                                                "Latest Epoch": last_metric.step,
                                                "Best Value": f"{best_metric.value:.4f}",
                                                "Best Epoch": best_metric.step,
                                            })
                                
                                if latest_metrics:
                                    metrics_df = pd.DataFrame(latest_metrics)
                                    st.dataframe(metrics_df, width="stretch", hide_index=True)
                                
                                # Show all epoch metrics table
                                st.markdown("**📊 All Epoch Metrics**")
                                trial_train_hist = client.get_metric_history(trial_run_id, f"train_loss_{trial_metric_prefix}")
                                trial_val_hist = client.get_metric_history(trial_run_id, f"val_loss_{trial_metric_prefix}")
                                
                                if trial_train_hist or trial_val_hist:
                                    epoch_data = {}
                                    max_epoch = 0
                                    
                                    for m in trial_train_hist:
                                        epoch = m.step
                                        max_epoch = max(max_epoch, epoch)
                                        if epoch not in epoch_data:
                                            epoch_data[epoch] = {}
                                        epoch_data[epoch]["train_loss"] = m.value
                                    
                                    for m in trial_val_hist:
                                        epoch = m.step
                                        max_epoch = max(max_epoch, epoch)
                                        if epoch not in epoch_data:
                                            epoch_data[epoch] = {}
                                        epoch_data[epoch]["val_loss"] = m.value
                                    
                                    # Add other metrics
                                    if task_type == "detection":
                                        for metric_name in ["f1_50_detection", "precision_50_detection", "recall_50_detection"]:
                                            hist = client.get_metric_history(trial_run_id, metric_name)
                                            for m in hist:
                                                epoch = m.step
                                                if epoch not in epoch_data:
                                                    epoch_data[epoch] = {}
                                                epoch_data[epoch][metric_name.replace("_detection", "")] = m.value
                                    else:
                                        for metric_name in ["pixel_acc_segmentation", "miou_segmentation"]:
                                            hist = client.get_metric_history(trial_run_id, metric_name)
                                            for m in hist:
                                                epoch = m.step
                                                if epoch not in epoch_data:
                                                    epoch_data[epoch] = {}
                                                epoch_data[epoch][metric_name.replace("_segmentation", "")] = m.value
                                    
                                    # Create DataFrame
                                    epoch_rows = []
                                    for epoch in sorted(epoch_data.keys()):
                                        row = {"Epoch": epoch}
                                        row.update(epoch_data[epoch])
                                        epoch_rows.append(row)
                                    
                                    if epoch_rows:
                                        epoch_df = pd.DataFrame(epoch_rows)
                                        st.dataframe(epoch_df, width="stretch", hide_index=True)
                                        
                                        # Download button
                                        csv = epoch_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="📥 Download Epoch Metrics as CSV",
                                            data=csv,
                                            file_name=f"trial_{trial_run.data.tags.get('mlflow.runName', idx)}_epoch_metrics.csv",
                                            mime="text/csv",
                                        )
                    
                    st.markdown("---")
                    st.markdown("#### ⚙️ Tuning Parameters")
                
                # Show run parameters
                st.markdown("#### ⚙️ Training Parameters")
                
                # Parametreleri kategorilere ayır
                params = run_info.data.params
                
                # Common parameters
                common_params = {}
                detection_params = {}
                segmentation_params = {}
                augmentation_params = {}
                early_stopping_params = {}
                overfitting_params = {}
                other_params = {}
                
                for key, value in params.items():
                    if key.startswith("common_"):
                        common_params[key.replace("common_", "")] = value
                    elif key.startswith("detection_"):
                        detection_params[key.replace("detection_", "")] = value
                    elif key.startswith("segmentation_"):
                        segmentation_params[key.replace("segmentation_", "")] = value
                    elif key.startswith("augmentation_"):
                        augmentation_params[key.replace("augmentation_", "")] = value
                    elif key.startswith("early_stopping_"):
                        early_stopping_params[key.replace("early_stopping_", "")] = value
                    elif key.startswith("overfitting_"):
                        overfitting_params[key.replace("overfitting_", "")] = value
                    else:
                        other_params[key] = value
                
                # Parametreleri göster
                col1, col2 = st.columns(2)
                
                def format_value(v):
                    """Format parameter values for better readability"""
                    if isinstance(v, bool):
                        return "✅ Yes" if v else "❌ No"
                    elif isinstance(v, (int, float)):
                        return f"{v:.4f}" if isinstance(v, float) else str(v)
                    else:
                        return str(v)
                
                with col1:
                    if common_params:
                        st.markdown("**📋 General Parameters**")
                        common_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(common_params.items())
                        ])
                        st.dataframe(common_df, hide_index=True, use_container_width=True)
                    
                    if task_type == "detection" and detection_params:
                        st.markdown("**🎯 Detection Parameters**")
                        det_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(detection_params.items())
                        ])
                        st.dataframe(det_df, hide_index=True, use_container_width=True)
                    
                    if task_type == "segmentation" and segmentation_params:
                        st.markdown("**🔍 Segmentation Parameters**")
                        seg_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(segmentation_params.items())
                        ])
                        st.dataframe(seg_df, hide_index=True, use_container_width=True)
                
                with col2:
                    if augmentation_params:
                        st.markdown("**🎨 Augmentation Parameters**")
                        aug_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(augmentation_params.items())
                        ])
                        st.dataframe(aug_df, hide_index=True, use_container_width=True)
                    
                    if early_stopping_params:
                        st.markdown("**⏸️ Early Stopping Parameters**")
                        es_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(early_stopping_params.items())
                        ])
                        st.dataframe(es_df, hide_index=True, use_container_width=True)
                    
                    if overfitting_params:
                        st.markdown("**🔍 Overfitting Detection Parameters**")
                        of_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(overfitting_params.items())
                        ])
                        st.dataframe(of_df, hide_index=True, use_container_width=True)
                    
                    if other_params:
                        st.markdown("**📌 Other Parameters**")
                        other_df = pd.DataFrame([
                            {"Parameter": k.replace("_", " ").title(), "Value": format_value(v)}
                            for k, v in sorted(other_params.items())
                        ])
                        st.dataframe(other_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")

                # --- Render artifacts early so we show images even if metrics are missing ---
                artifact_paths = _list_all_artifacts(run_id)
                image_artifacts = [p for p in artifact_paths if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg"}]

                if task_type == "detection":
                    curve_candidates = [
                        p for p in image_artifacts if any(curve in Path(p).name.lower() for curve in ["f1_curve", "pr_curve", "p_curve", "r_curve"])
                    ]
                    train_batch_candidates = [p for p in image_artifacts if "train_batch" in Path(p).name.lower()]
                    val_batch_candidates = [p for p in image_artifacts if "val_batch" in Path(p).name.lower()]
                    plot_candidates = [p for p in image_artifacts if Path(p).name.lower() in {"results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "labels.jpg", "labels_correlogram.jpg"}]
                    preview_candidates = [
                        p for p in image_artifacts if any(k in Path(p).name.lower() for k in ["prediction", "preview"]) and "batch" not in Path(p).name.lower()
                    ]

                    if plot_candidates or curve_candidates or train_batch_candidates or val_batch_candidates or preview_candidates:
                        st.markdown("---")
                    if plot_candidates:
                        _render_images(run_id, "#### 📊 Training Summary", plot_candidates, max_images=8)
                    if curve_candidates:
                        _render_images(run_id, "#### 📈 Metric Curves", curve_candidates, max_images=8)
                    if train_batch_candidates:
                        _render_images(run_id, "#### 🎨 Train Batch Visualizations", train_batch_candidates, max_images=8)
                    if val_batch_candidates:
                        _render_images(run_id, "#### ✅ Validation Batch Visualizations", val_batch_candidates, max_images=8)
                    if preview_candidates:
                        _render_images(run_id, "#### 📸 Sample Predictions", preview_candidates, max_images=12)
                else:
                    plot_candidates = [p for p in image_artifacts if Path(p).name.lower() in {"results.png", "confusion_matrix.png", "labels.jpg"}]
                    preview_candidates = [p for p in image_artifacts if any(k in Path(p).name.lower() for k in ["prediction", "preview"])]
                    if plot_candidates or preview_candidates:
                        st.markdown("---")
                    if plot_candidates:
                        _render_images(run_id, "#### 🖼️ Training Plots", plot_candidates, max_images=6)
                    if preview_candidates:
                        _render_images(run_id, "#### 📸 Sample Predictions", preview_candidates, max_images=12)

                # --- Metrics and live graphs ---
                train_hist = client.get_metric_history(run_id, metric_train)
                val_hist = client.get_metric_history(run_id, metric_val)

                if train_hist:
                    train_hist = sorted(train_hist, key=lambda x: x.step)
                if val_hist:
                    val_hist = sorted(val_hist, key=lambda x: x.step)

                if train_hist or val_hist:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        last_train = train_hist[-1].value if train_hist else None
                        best_train = min(train_hist, key=lambda x: x.value) if train_hist else None
                        best_train_delta = (
                            f"Best: {best_train.value:.4f} (Epoch {best_train.step})"
                            if best_train and best_train.value != last_train
                            else None
                        )
                        st.metric("Train Loss", f"{last_train:.4f}" if last_train else "N/A", delta=best_train_delta)

                    with col2:
                        last_val = val_hist[-1].value if val_hist else None
                        best_val = min(val_hist, key=lambda x: x.value) if val_hist else None
                        best_val_delta = (
                            f"Best: {best_val.value:.4f} (Epoch {best_val.step})" if best_val and best_val.value != last_val else None
                        )
                        st.metric("Val Loss", f"{last_val:.4f}" if last_val else "N/A", delta=best_val_delta)

                    with col3:
                        max_epoch = max(train_hist[-1].step if train_hist else 0, val_hist[-1].step if val_hist else 0)
                        total_epochs = run_info.data.params.get("common_epochs", "?")
                        st.metric("Epoch", f"{max_epoch}/{total_epochs}")

                    with col4:
                        status = run_info.info.status
                        status_emoji = "🟢" if status == "FINISHED" else "🟡" if status == "RUNNING" else "🔴"
                        st.metric("Status", f"{status_emoji} {status}")

                    if train_hist or val_hist:
                        fig, ax = plt.subplots(figsize=(10, 5))

                        if train_hist:
                            train_sorted = sorted(train_hist, key=lambda x: x.step)
                            train_steps = [m.step for m in train_sorted]
                            train_values = [m.value for m in train_sorted]
                            ax.plot(train_steps, train_values, label="Train Loss", marker="o", linewidth=2)

                        if val_hist:
                            val_sorted = sorted(val_hist, key=lambda x: x.step)
                            val_steps = [m.step for m in val_sorted]
                            val_values = [m.value for m in val_sorted]
                            ax.plot(val_steps, val_values, label="Val Loss", marker="s", linestyle="--", linewidth=2)

                        ax.set_xlabel("Epoch", fontsize=12)
                        ax.set_ylabel("Loss", fontsize=12)
                        ax.set_title(f"Live Loss Chart - {run_name}", fontsize=14)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        st.markdown("#### 📊 All Metrics")
                        all_metrics_fig, all_metrics_ax = plt.subplots(figsize=(12, 6))

                        if train_hist:
                            train_sorted = sorted(train_hist, key=lambda x: x.step)
                            all_metrics_ax.plot(
                                [m.step for m in train_sorted],
                                [m.value for m in train_sorted],
                                label="Train Loss",
                                marker="o",
                                linewidth=2,
                                color="#1f77b4",
                            )

                        if val_hist:
                            val_sorted = sorted(val_hist, key=lambda x: x.step)
                            all_metrics_ax.plot(
                                [m.step for m in val_sorted],
                                [m.value for m in val_sorted],
                                label="Val Loss",
                                marker="s",
                                linestyle="--",
                                linewidth=2,
                                color="#ff7f0e",
                            )

                        if task_type == "detection":
                            metric_colors = ["#2ca02c", "#d62728", "#9467bd"]
                            for idx, (metric_name, display_name, style) in enumerate(
                                [
                                    ("precision_50_detection", "Precision@0.5", "-"),
                                    ("recall_50_detection", "Recall@0.5", "-"),
                                    ("f1_50_detection", "F1@0.5", "-"),
                                ]
                            ):
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    hist_sorted = sorted(hist, key=lambda x: x.step)
                                    all_metrics_ax.plot(
                                        [m.step for m in hist_sorted],
                                        [m.value for m in hist_sorted],
                                        label=display_name,
                                        marker="^",
                                        linestyle=style,
                                        linewidth=2,
                                        color=metric_colors[idx % len(metric_colors)],
                                        zorder=2,
                                    )
                        else:
                            metric_colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
                            base_metrics = [
                                ("pixel_acc_segmentation", "Pixel Accuracy", "-"),
                                ("miou_segmentation", "mIoU", "-")
                            ]

                            # Discover per-class IoU metrics dynamically
                            class_metric_keys = [
                                k for k in run_info.data.metrics.keys() if k.startswith("iou_class_")
                            ]
                            for key in sorted(class_metric_keys):
                                name = key.replace("iou_class_", "").replace("_segmentation", "")
                                display = f"IoU {name.replace('_', ' ')}"
                                base_metrics.append((key, display, "--"))
                            
                            for idx, (metric_name, display_name, style) in enumerate(base_metrics):
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    hist_sorted = sorted(hist, key=lambda x: x.step)
                                    all_metrics_ax.plot(
                                        [m.step for m in hist_sorted],
                                        [m.value for m in hist_sorted],
                                        label=display_name,
                                        marker="^",
                                        linestyle=style,
                                        linewidth=2,
                                        color=metric_colors[idx % len(metric_colors)],
                                        zorder=2,
                                    )

                        all_metrics_ax.set_xlabel("Epoch", fontsize=12)
                        all_metrics_ax.set_ylabel("Value", fontsize=12)
                        all_metrics_ax.set_title(f"All Metrics - {run_name}", fontsize=14)
                        all_metrics_ax.legend(loc="best", fontsize=10)
                        all_metrics_ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(all_metrics_fig)

                        st.markdown("#### 📈 Metric Summary (Latest and Best Values)")
                        latest_metrics = []

                        def get_best_metric(hist, is_loss=True):
                            if not hist:
                                return None, None
                            hist_sorted = sorted(hist, key=lambda x: x.step)
                            if is_loss:
                                best = min(hist_sorted, key=lambda x: x.value)
                            else:
                                best = max(hist_sorted, key=lambda x: x.value)
                            return best, hist_sorted[-1]

                        if train_hist:
                            best_train, last_train = get_best_metric(train_hist, is_loss=True)
                            latest_metrics.append(
                                {
                                    "Metric": "Train Loss",
                                    "Latest Value": f"{last_train.value:.4f}",
                                    "Latest Epoch": last_train.step,
                                    "Best Value": f"{best_train.value:.4f}",
                                    "Best Epoch": best_train.step,
                                }
                            )

                        if val_hist:
                            best_val, last_val = get_best_metric(val_hist, is_loss=True)
                            latest_metrics.append(
                                {
                                    "Metric": "Val Loss",
                                    "Latest Value": f"{last_val.value:.4f}",
                                    "Latest Epoch": last_val.step,
                                    "Best Value": f"{best_val.value:.4f}",
                                    "Best Epoch": best_val.step,
                                }
                            )

                        if task_type == "detection":
                            for metric_name, display_name in [
                                ("precision_50_detection", "Precision@0.5"),
                                ("recall_50_detection", "Recall@0.5"),
                                ("f1_50_detection", "F1@0.5"),
                            ]:
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    best_metric, last_metric = get_best_metric(hist, is_loss=False)
                                    latest_metrics.append(
                                        {
                                            "Metric": display_name,
                                            "Latest Value": f"{last_metric.value:.4f}",
                                            "Latest Epoch": last_metric.step,
                                            "Best Value": f"{best_metric.value:.4f}",
                                            "Best Epoch": best_metric.step,
                                        }
                                    )
                        else:
                            base_metrics = [
                                ("pixel_acc_segmentation", "Pixel Accuracy"),
                                ("miou_segmentation", "mIoU"),
                            ]

                            # Add per-class IoU metrics dynamically
                            class_metric_keys = [
                                k for k in run_info.data.metrics.keys() if k.startswith("iou_class_")
                            ]
                            for key in sorted(class_metric_keys):
                                name = key.replace("iou_class_", "").replace("_segmentation", "")
                                display = f"IoU {name.replace('_', ' ')}"
                                base_metrics.append((key, display))
                            
                            for metric_name, display_name in base_metrics:
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    best_metric, last_metric = get_best_metric(hist, is_loss=False)
                                    latest_metrics.append(
                                        {
                                            "Metric": display_name,
                                            "Latest Value": f"{last_metric.value:.4f}",
                                            "Latest Epoch": last_metric.step,
                                            "Best Value": f"{best_metric.value:.4f}",
                                            "Best Epoch": best_metric.step,
                                        }
                                    )

                        if latest_metrics:
                            metrics_df = pd.DataFrame(latest_metrics)
                            st.dataframe(metrics_df, width="stretch", hide_index=True)
                        
                        # All epoch metrics table
                        st.markdown("#### 📊 All Epoch Metrics")
                        st.caption("Use the table below to view all metric values per epoch")
                        
                        # Epoch sayısını bul
                        max_epoch = 0
                        if train_hist:
                            max_epoch = max(max_epoch, max(m.step for m in train_hist))
                        if val_hist:
                            max_epoch = max(max_epoch, max(m.step for m in val_hist))
                        
                        # Tüm metrikleri epoch bazında topla
                        metric_dicts = {}
                        
                        # Train Loss
                        if train_hist:
                            for m in train_hist:
                                epoch = m.step
                                if epoch not in metric_dicts:
                                    metric_dicts[epoch] = {}
                                metric_dicts[epoch]["Train Loss"] = m.value
                        
                        # Val Loss
                        if val_hist:
                            for m in val_hist:
                                epoch = m.step
                                if epoch not in metric_dicts:
                                    metric_dicts[epoch] = {}
                                metric_dicts[epoch]["Val Loss"] = m.value
                        
                        # Task-specific metrics
                        if task_type == "detection":
                            for metric_name, display_name in [
                                ("precision_50_detection", "Precision@0.5"),
                                ("recall_50_detection", "Recall@0.5"),
                                ("f1_50_detection", "F1@0.5"),
                            ]:
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    for m in hist:
                                        epoch = m.step
                                        if epoch not in metric_dicts:
                                            metric_dicts[epoch] = {}
                                        metric_dicts[epoch][display_name] = m.value
                                        max_epoch = max(max_epoch, epoch)
                        else:
                            # Segmentation metrics
                            base_metrics = [
                                ("pixel_acc_segmentation", "Pixel Accuracy"),
                                ("miou_segmentation", "mIoU"),
                            ]
                            
                            # Per-class IoU metrics
                            class_metric_keys = [
                                k for k in run_info.data.metrics.keys() if k.startswith("iou_class_")
                            ]
                            for key in sorted(class_metric_keys):
                                name = key.replace("iou_class_", "").replace("_segmentation", "")
                                display = f"IoU {name.replace('_', ' ')}"
                                base_metrics.append((key, display))
                            
                            for metric_name, display_name in base_metrics:
                                hist = client.get_metric_history(run_id, metric_name)
                                if hist:
                                    for m in hist:
                                        epoch = m.step
                                        if epoch not in metric_dicts:
                                            metric_dicts[epoch] = {}
                                        metric_dicts[epoch][display_name] = m.value
                                        max_epoch = max(max_epoch, epoch)
                        
                        # Create DataFrame
                        epoch_rows = []
                        for epoch in range(max_epoch + 1):
                            row = {"Epoch": epoch}
                            if epoch in metric_dicts:
                                row.update(metric_dicts[epoch])
                            epoch_rows.append(row)
                        
                        if epoch_rows:
                            epoch_df = pd.DataFrame(epoch_rows)
                            # Sadece veri olan epoch'ları göster
                            epoch_df = epoch_df[epoch_df.iloc[:, 1:].notna().any(axis=1)]
                            
                            # Sıralama ve formatlama
                            epoch_df = epoch_df.sort_values("Epoch").reset_index(drop=True)
                            
                            # Sayısal sütunları formatla
                            numeric_cols = epoch_df.select_dtypes(include=['float64', 'int64']).columns
                            for col in numeric_cols:
                                if col != "Epoch":
                                    epoch_df[col] = epoch_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                            
                            # Show table
                            st.dataframe(
                                epoch_df,
                                width="stretch",
                                hide_index=True,
                                use_container_width=True,
                            )
                            
                            # Download button
                            csv = epoch_df.to_csv(index=False)
                            # Dosya adını temizle (özel karakterleri kaldır)
                            safe_filename = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name))
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"{safe_filename}_all_epochs_metrics.csv",
                                mime="text/csv",
                            )
        else:
            st.info(f"🔍 {len(table_selected_run_ids)} runs selected. Live metrics for each run are shown above.")

    if auto_refresh and refresh_interval:
        placeholder = st.empty()
        with placeholder.container():
            st.info(f"🔄 Will auto-refresh in {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        placeholder.empty()
        st.rerun()
