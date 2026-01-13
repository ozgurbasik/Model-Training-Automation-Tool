import mlflow

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("autonomous_vehicle")

if exp:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.task_type = 'segmentation'",
        max_results=1
    )
    
    if runs:
        run = runs[0]
        print(f"Run ID: {run.info.run_id[:8]}")
        print(f"\nAll metrics for this run:")
        
        # Get all metrics
        all_metrics = {}
        for key in run.data.metrics.keys():
            history = client.get_metric_history(run.info.run_id, key)
            if history:
                last_value = history[-1].value
                all_metrics[key] = last_value
        
        # Sort and print
        for key in sorted(all_metrics.keys()):
            print(f"  {key}: {all_metrics[key]:.4f}")
