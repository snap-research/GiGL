import os
from typing import Optional

import kfp
import kfp.containers
import kfp.gcp
from kfp.dsl import PipelineParam
from kfp.dsl._container_op import ContainerOp

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.orchestration.kubeflow.utils.glt_backend import (
    check_glt_backend_eligibility_component,
)
from gigl.orchestration.kubeflow.utils.log_metrics import log_eval_metrics_to_ui
from gigl.orchestration.kubeflow.utils.resource import add_task_resource_requirements
from gigl.src.common.constants.components import GiGLComponents

COMPONENTS_BASE_PATH = os.path.join(
    local_fs_constants.get_gigl_root_directory(),
    "orchestration",
    "kubeflow",
)

logger = Logger()

SPECED_COMPONENTS = [
    GiGLComponents.ConfigValidator.value,
    GiGLComponents.ConfigPopulator.value,
    GiGLComponents.SubgraphSampler.value,
    GiGLComponents.DataPreprocessor.value,
    GiGLComponents.SplitGenerator.value,
    GiGLComponents.Inferencer.value,
    GiGLComponents.PostProcessor.value,
    GiGLComponents.Trainer.value,
]

speced_component_root: LocalUri = LocalUri.join(COMPONENTS_BASE_PATH, "components")
speced_component_op_dict = {
    component: kfp.components.load_component_from_file(
        LocalUri.join(speced_component_root, component, "component.yaml").uri
    )
    for component in SPECED_COMPONENTS
}


def generate_component_task(
    component: str,
    job_name: str,
    uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    start_at: Optional[str] = None,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    component_task_op: ContainerOp

    if component == GiGLComponents.ConfigPopulator.value:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            template_uri=uri,
            resource_config_uri=resource_config_uri,
        )
    elif component == GiGLComponents.ConfigValidator.value:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=uri,
            start_at=start_at,
            resource_config_uri=resource_config_uri,
            stop_after=stop_after,
        )
    elif component == GiGLComponents.Trainer.value:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            config_uri=uri,
            resource_config_uri=resource_config_uri,
            cpu_docker_uri=common_pipeline_component_configs.cpu_container_image,
            cuda_docker_uri=common_pipeline_component_configs.cuda_container_image,
        )
    elif component == GiGLComponents.DataPreprocessor.value:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=uri,
            resource_config_uri=resource_config_uri,
            custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
        )
    elif component == GiGLComponents.Inferencer.value:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=uri,
            resource_config_uri=resource_config_uri,
            custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
            cpu_docker_uri=common_pipeline_component_configs.cpu_container_image,
            cuda_docker_uri=common_pipeline_component_configs.cuda_container_image,
        )
    else:
        component_task_op = speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=uri,
            resource_config_uri=resource_config_uri,
        )
    add_task_resource_requirements(
        task=component_task_op,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    return component_task_op


def generate_pipeline(
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
):
    """
    Generates a KFP pipeline definition for GiGL.
    Args:
        common_pipeline_component_configs (CommonPipelineComponentConfigs): Shared configuration between components.

    Returns:
        An @kfp.dsl.pipeline decorated function to generated a pipeline.
    """
    if (
        common_pipeline_component_configs.additional_job_args
        and GiGLComponents.SubgraphSampler
        not in common_pipeline_component_configs.additional_job_args
    ):
        raise ValueError(
            f"Only additional args for Subgraph Sampler are supported. Received {common_pipeline_component_configs.additional_job_args}"
        )

    @kfp.dsl.pipeline(
        name="GiGL_Pipeline",
        description="GiGL Pipeline",
    )
    def pipeline(
        job_name,
        template_or_frozen_config_uri,
        resource_config_uri,
        start_at=GiGLComponents.ConfigPopulator.value,
        stop_after=None,
    ):
        validation_check_task = generate_component_task(
            component=GiGLComponents.ConfigValidator.value,
            job_name=job_name,
            uri=template_or_frozen_config_uri,
            start_at=start_at,
            stop_after=stop_after,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
        )

        # TODO (mkolodner-sc): Update method for specifying glt_backend once long-term alignment is reached
        check_glt_backend_eligibility_component_generator = (
            kfp.components.func_to_container_op(
                check_glt_backend_eligibility_component,
                base_image=common_pipeline_component_configs.cpu_container_image,
            )
        )
        check_glt_backend_eligibility_container_op: ContainerOp = (
            check_glt_backend_eligibility_component_generator(
                task_config_uri=template_or_frozen_config_uri
            )
        )
        check_glt_backend_eligibility_container_op.set_display_name(
            name="Check whether to use GLT Backend"
        )
        should_use_glt_runtime_param: PipelineParam = (
            check_glt_backend_eligibility_container_op.output
        )

        with kfp.dsl.Condition(start_at == GiGLComponents.ConfigPopulator.value):
            config_populator_task = create_config_populator_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                should_use_glt_runtime_param=should_use_glt_runtime_param,
                stop_after=stop_after,
            )
            config_populator_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.DataPreprocessor.value):
            data_preprocessor_task = create_data_preprocessor_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
                should_use_glt_runtime_param=should_use_glt_runtime_param,
            )
            data_preprocessor_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.SubgraphSampler.value):
            subgraph_sampler_task = create_subgraph_sampler_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            subgraph_sampler_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.SplitGenerator.value):
            split_generator_task = create_split_generator_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            split_generator_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.Trainer.value):
            trainer_task = create_trainer_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            trainer_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.Inferencer.value):
            inferencer_task = create_inferencer_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            inferencer_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.PostProcessor.value):
            post_processor_task = create_post_processor_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            post_processor_task.after(validation_check_task)

    return pipeline


def create_config_populator_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    should_use_glt_runtime_param: PipelineParam,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    config_populator_task = generate_component_task(
        component=GiGLComponents.ConfigPopulator.value,
        job_name=job_name,
        uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
        stop_after=stop_after,
    )
    frozen_gbml_config_uri = config_populator_task.outputs["frozen_gbml_config_uri"]

    with kfp.dsl.Condition(stop_after != GiGLComponents.ConfigPopulator.value):
        data_preprocessor_task = create_data_preprocessor_task_op(
            job_name=job_name,
            task_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            should_use_glt_runtime_param=should_use_glt_runtime_param,
            stop_after=stop_after,
        )
        data_preprocessor_task.after(config_populator_task)
    return config_populator_task


def create_data_preprocessor_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    should_use_glt_runtime_param: PipelineParam,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    data_preprocessor_task = generate_component_task(
        component=GiGLComponents.DataPreprocessor.value,
        job_name=job_name,
        uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.DataPreprocessor.value):
        with kfp.dsl.Condition(should_use_glt_runtime_param == False):
            subgraph_sampler_task = create_subgraph_sampler_task_op(
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            subgraph_sampler_task.after(data_preprocessor_task)
        # If we are using the GLT runtime, we skip the subgraph sampler and split generator
        # and go straight to the GLT trainer
        with kfp.dsl.Condition(should_use_glt_runtime_param == True):
            glt_trainer_task = create_trainer_task_op(
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            glt_trainer_task.after(data_preprocessor_task)

    return data_preprocessor_task


def create_subgraph_sampler_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    subgraph_sampler_task = speced_component_op_dict["subgraph_sampler"](
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
        **(
            common_pipeline_component_configs.additional_job_args.get(
                GiGLComponents.SubgraphSampler
            )
            or {}
        ),
    )
    add_task_resource_requirements(
        task=subgraph_sampler_task,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.SubgraphSampler.value):
        split_generator_task = create_split_generator_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        split_generator_task.after(subgraph_sampler_task)

    return subgraph_sampler_task


def create_split_generator_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    split_generator_task: ContainerOp
    split_generator_task = speced_component_op_dict["split_generator"](
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
    )
    add_task_resource_requirements(
        task=split_generator_task,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.SplitGenerator.value):
        trainer_task = create_trainer_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        trainer_task.after(split_generator_task)

    return split_generator_task


def create_inferencer_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    inferencer_task = generate_component_task(
        component=GiGLComponents.Inferencer.value,
        job_name=job_name,
        uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.Inferencer.value):
        post_processor_task = create_post_processor_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
        )
        post_processor_task.after(inferencer_task)

    return inferencer_task


def create_trainer_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    trainer_task = generate_component_task(
        component=GiGLComponents.Trainer.value,
        job_name=job_name,
        uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    log_metrics_op = kfp.components.func_to_container_op(
        log_eval_metrics_to_ui,
        base_image=common_pipeline_component_configs.cpu_container_image,
    )
    log_metrics_task_op: ContainerOp = log_metrics_op(
        task_config_uri=task_config_uri, component=GiGLComponents.Trainer.value
    )
    log_metrics_task_op.set_display_name(name="Log Trainer Eval Metrics")
    log_metrics_task_op.after(trainer_task)

    with kfp.dsl.Condition(stop_after != GiGLComponents.Trainer.value):
        inference_task = create_inferencer_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        inference_task.after(trainer_task)
    return trainer_task


def create_post_processor_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> ContainerOp:
    post_processor_task = generate_component_task(
        component=GiGLComponents.PostProcessor.value,
        job_name=job_name,
        uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )
    # Log post processor eval metrics
    log_metrics_op = kfp.components.func_to_container_op(
        log_eval_metrics_to_ui,
        base_image=common_pipeline_component_configs.cpu_container_image,
    )
    log_metrics_task_op: ContainerOp = log_metrics_op(
        task_config_uri=task_config_uri,
        component=GiGLComponents.PostProcessor.value,
    )
    log_metrics_task_op.set_display_name(name="Log PostProcessor Eval Metrics")
    log_metrics_task_op.after(post_processor_task)
    return post_processor_task
