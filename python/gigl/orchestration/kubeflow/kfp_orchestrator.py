import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from kfp.compiler import Compiler

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.services.kfp import KFPService
from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.env.dep_constants import (
    GIGL_DATAFLOW_IMAGE,
    GIGL_SRC_IMAGE_CPU,
    GIGL_SRC_IMAGE_CUDA,
)
from gigl.orchestration.kubeflow.kfp_pipeline import generate_pipeline
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.time import current_formatted_datetime

logger = Logger()


DEFAULT_PIPELINE_VERSION_NAME = (
    f"gigl-pipeline-version-at-{current_formatted_datetime()}"
)

GIGL_PIPELINE_BUDLE_PATH = LocalUri.join(
    local_fs_constants.get_project_root_directory(),
    "build",
    f"gigl_pipeline_gnn.tar.gz",
)

DEFAULT_START_AT_COMPONENT = "config_populator"


@dataclass
class KfpEnvMetadata:
    kfp_host: str
    k8_sa: str
    experiment_id: str
    pipeline_id: str

    def __repr__(self) -> str:
        return (
            f"KfpEnvMetadata("
            f"kfp_host={self.kfp_host}, "
            f"k8_sa={self.k8_sa}, "
            f"experiment_id={self.experiment_id}, "
            f"pipeline_id={self.pipeline_id})"
            f")"
        )


class KfpOrchestrator:
    """
    Orchestration of Kubeflow Pipelines for GiGL.
    Args:
        kfp_metadata (Optional[KfpEnvMetadata]): KFP environment metadata. If not provided, it will be loaded from the environment.
        env_path (Optional[str]): Path to the environment file containing KFP metadata. Default checks in the current directory.
    Methods:
        compile: Compiles the Kubeflow pipeline.
        run: Runs the Kubeflow pipeline.
        upload: Uploads the pipeline to KFP.
        wait_for_completion: Waits for the pipeline run to complete.
    """

    def __init__(
        self,
        kfp_metadata: Optional[KfpEnvMetadata] = None,
        env_path: Optional[str] = None,
    ):
        if kfp_metadata:
            self.kfp_metadata = kfp_metadata
        else:
            self.kfp_metadata = self._load_kfp_metadata(env_path=env_path)
        self.kfp_service = KFPService(
            kfp_host=self.kfp_metadata.kfp_host,
            k8_sa=self.kfp_metadata.k8_sa,
        )

    @staticmethod
    def _load_kfp_metadata(env_path: Optional[str] = None) -> KfpEnvMetadata:
        load_dotenv(dotenv_path=env_path)

        return KfpEnvMetadata(
            kfp_host=os.getenv("KFP_HOST", "default_host"),
            k8_sa=os.getenv("K8_SA", "default_sa"),
            experiment_id=os.getenv("EXPERIMENT_ID", "default_experiment_id"),
            pipeline_id=os.getenv("PIPELINE_ID", "default_pipeline_id"),
        )

    @classmethod
    def compile(
        cls,
        cuda_container_image: str,
        cpu_container_image: str,
        dataflow_container_image: str,
        additional_job_args: Optional[dict[GiGLComponents, dict[str, str]]] = None,
    ) -> LocalUri:
        """
        Compiles the GiGL Kubeflow pipeline.
        Args:
            cuda_container_image (str): Container image for CUDA (see: containers/Dockerfile.cuda).
            cpu_container_image (str): Container image for CPU.
            dataflow_container_image (str): Container image for Dataflow.
            additional_job_args: Optional additional arguements to be passed into components, by component.
        """
        pipeline_bundle_path: LocalUri = GIGL_PIPELINE_BUDLE_PATH
        Path(pipeline_bundle_path.uri).parent.mkdir(parents=True, exist_ok=True)

        common_pipeline_component_configs = CommonPipelineComponentConfigs(
            cuda_container_image=cuda_container_image,
            cpu_container_image=cpu_container_image,
            dataflow_container_image=dataflow_container_image,
            additional_job_args=additional_job_args or {},
        )

        Compiler().compile(
            generate_pipeline(
                common_pipeline_component_configs=common_pipeline_component_configs,
            ),
            pipeline_bundle_path.uri,
        )

        logger.info(f"Compiled Kubeflow pipeline to {pipeline_bundle_path.uri}")

        return pipeline_bundle_path

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        start_at: str = DEFAULT_START_AT_COMPONENT,
        stop_after: Optional[str] = None,
        cuda_container_image: str = GIGL_SRC_IMAGE_CUDA,
        cpu_container_image: str = GIGL_SRC_IMAGE_CPU,
        dataflow_container_image: str = GIGL_DATAFLOW_IMAGE,
        compile: bool = True,
        additional_job_args: Optional[dict[GiGLComponents, dict[str, str]]] = None,
    ) -> str:
        if compile:
            pipeline_budle_path = self.compile(
                cuda_container_image=cuda_container_image,
                cpu_container_image=cpu_container_image,
                dataflow_container_image=dataflow_container_image,
                additional_job_args=additional_job_args,
            )

        run_keyword_args = {
            "job_name": applied_task_identifier,
            "start_at": start_at,
            "template_or_frozen_config_uri": task_config_uri.uri,
            "resource_config_uri": resource_config_uri.uri,
        }
        if stop_after is not None:
            run_keyword_args["stop_after"] = stop_after

        logger.info(f"Running pipeline with args: {run_keyword_args}")
        run_id = self.kfp_service.run_pipeline(
            pipeline_bundle_path=str(pipeline_budle_path),
            experiment_id=self.kfp_metadata.experiment_id,
            run_name=applied_task_identifier,
            run_keyword_args=run_keyword_args,
        )

        return run_id

    def upload(self, pipeline_version_name: str = DEFAULT_PIPELINE_VERSION_NAME) -> str:
        logger.info(
            f"Uploading pipeline version: {pipeline_version_name} to pipeline id: {self.kfp_metadata.pipeline_id}"
        )
        upload_url = self.kfp_service.upload_pipeline_version(
            pipeline_bundle_path=str(GIGL_PIPELINE_BUDLE_PATH),
            pipeline_id=self.kfp_metadata.pipeline_id,
            pipeline_version_name=pipeline_version_name,
        )

        return upload_url

    def wait_for_completion(self, run_id: str):
        self.kfp_service.wait_for_run_completion(run_id=run_id)
