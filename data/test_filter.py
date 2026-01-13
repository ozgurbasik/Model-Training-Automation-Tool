import mlflow

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("autonomous_vehicle")

if exp:
    # Test different filter strings
    print("Testing filter: params.task_type = 'segmentation'")
    runs1 = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.task_type = 'segmentation'"
    )
    print(f"Found {len(runs1)} runs with params.task_type = 'segmentation'")
    
    print("\nTesting filter: params.task_type = 'detection'")
    runs2 = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.task_type = 'detection'"
    )
    print(f"Found {len(runs2)} runs with params.task_type = 'detection'")
    
    print("\nAll runs without filter:")
    runs3 = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"Found {len(runs3)} total runs")
    
    # Check a sample run's params
    if runs3:
        sample_run = runs3[0]
        print(f"\nSample run params: {sample_run.data.params}")
else:
    print("No experiment found")
