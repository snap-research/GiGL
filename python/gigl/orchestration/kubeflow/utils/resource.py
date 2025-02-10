from kfp.dsl._container_op import ContainerOp

from gigl.common.types.resource_config import CommonPipelineComponentConfigs


def add_task_resource_requirements(
    task: ContainerOp,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
):
    """
    Adds resource requirements to a the Kubeflow Pipeline (KFP) Task (ContainerOp)

    Args:
        task (ContainerOp): The task to add resource requirements to.
        common_pipeline_component_configs (CommonPipelineComponentConfigs): The common pipeline component configurations.

    Returns:
        None
    """
    DEFAULT_CPU_REQUEST = "4"
    DEFAULT_MEMORY_REQUEST = "1Gi"
    # default to cpu image, overwrite later as needed
    task.container.image = common_pipeline_component_configs.cpu_container_image
    task.container.set_cpu_request(DEFAULT_CPU_REQUEST)
    task.container.set_memory_request(DEFAULT_MEMORY_REQUEST)
