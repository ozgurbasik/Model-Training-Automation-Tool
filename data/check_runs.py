import mlflow

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("autonomous_vehicle")

if exp:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"Total runs: {len(runs)}")
    
    for i, r in enumerate(runs[:20]):
        task_type = r.data.params.get("task_type")
        run_name = r.data.tags.get("mlflow.runName", "N/A")
        print(f"Run {i}: task_type='{task_type}', run_name='{run_name}', run_id={r.info.run_id[:8]}")
else:
    print("No experiment found")
