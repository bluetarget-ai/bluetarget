import mlfoundry as mlf

client = mlf.get_client()

inference_data = client.get_inference_dataset(
    model_fqn="",
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    actual_value_required=True,
)
