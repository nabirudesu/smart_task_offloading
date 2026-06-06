import ray

ray.shutdown()  # in case it was already running
ray.init(include_dashboard=False)
print("Ray initialized:", ray.is_initialized())
