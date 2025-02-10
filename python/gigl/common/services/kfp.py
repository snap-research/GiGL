import json
import os
from enum import Enum
from typing import Dict, Optional

import kfp  # type: ignore
import requests
from kfp._auth import get_gcp_access_token
from kfp_server_api.models.api_list_experiments_response import (
    ApiListExperimentsResponse,
)
from kfp_server_api.models.api_list_runs_response import ApiListRunsResponse
from kfp_server_api.models.api_run import ApiRun
from kfp_server_api.models.api_run_detail import ApiRunDetail

from gigl.common.logger import Logger
from gigl.common.types.wrappers.kfp_api import ApiRunDetailWrapper
from gigl.common.utils.func_tools import lru_cache

logger = Logger()


class _KfpApiFilterOperations(Enum):
    # See: https://github.com/kubeflow/pipelines/blob/ed9a5abe3a69c5e9269a375d334df16423ed5ca1/backend/api/v1beta1/filter.proto#L27
    UNKNOWN = 0
    EQUALS = 1
    NOT_EQUALS = 2
    GREATER_THAN = 3
    GREATER_THAN_EQUALS = 5
    LESS_THAN = 6
    LESS_THAN_EQUALS = 7
    IN = 8
    IS_SUBSTRING = 9


class KFPService:
    """
    A service class that provides methods to interact with Kubeflow Pipelines (KFP).
    """

    def __init__(self, kfp_host: str, k8_sa: str) -> None:
        """
        Initializes a KFPService object.

        Args:
            kfp_host (str): The host URL of the KFP instance.
            k8_sa (str): The service account associated with the KFP instance.
        """
        self.kfp_host = kfp_host
        self.kfp_client = kfp.Client(host=self.kfp_host)
        self.k8_sa = k8_sa

    def __get_auth_header(self) -> Dict[str, str]:
        assert self.kfp_host.startswith(
            "https://"
        ), "Lets not send our tokens over unencrypted connections"
        assert self.kfp_host.endswith(
            ".pipelines.googleusercontent.com"
        ), "Only GCP hosted KFP instances are supported"
        header = {
            "Authorization": f"Bearer {get_gcp_access_token()}",
        }
        return header

    def get_experiment_name_by_id(self, experiment_id: str) -> str:
        """
        Retrieves the name of a Kubeflow Pipelines experiment given its ID.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            str: The name of the experiment.
        """
        return self.kfp_client.get_experiment(experiment_id=experiment_id).name

    def run_pipeline(
        self,
        pipeline_bundle_path: str,
        run_name: str,
        experiment_id: str,
        run_keyword_args: Dict[str, str],
    ) -> str:
        """
        Runs a pipeline using the KFP instance.

        Args:
            pipeline_bundle_path (str): The path to the pipeline bundle file.
            run_name (str): The name of the run.
            experiment_id (str): The ID of the experiment.
            run_keyword_args (Dict[str, str]): The keyword arguments for the run.

        Returns:
            str: The ID of the run.
        """
        if not os.path.isfile(pipeline_bundle_path):
            raise RuntimeError(
                f"Pipeline bundle file does not exist at {pipeline_bundle_path}"
            )

        experiment_name = (
            self.get_experiment_name_by_id(experiment_id) if experiment_id else None
        )

        pipeline_res = self.kfp_client.create_run_from_pipeline_package(
            pipeline_file=pipeline_bundle_path,
            arguments=run_keyword_args,
            run_name=run_name,
            experiment_name=experiment_name,
            service_account=self.k8_sa,
        )
        logger.info(
            f"Created run @ {self.kfp_host}/#/runs/details/{pipeline_res.run_id}"
        )

        return pipeline_res.run_id

    def upload_pipeline_version(
        self, pipeline_bundle_path: str, pipeline_id: str, pipeline_version_name: str
    ) -> str:
        """
        Uploads the pipeline to the Kubeflow Pipelines cluster.

        Args:
            pipeline_bundle_path (str): The path to the pipeline bundle file.
            pipeline_id (str): The ID of the pipeline.
            pipeline_version_name (str): The name of the pipeline version.

        Returns:
            str: The URL of the pipeline on the Kubeflow Pipelines cluster.
        """
        if not os.path.isfile(pipeline_bundle_path):
            raise RuntimeError(
                f"Pipeline bundle file does not exist at {pipeline_bundle_path}"
            )

        pipeline_version = self.kfp_client.pipeline_uploads.upload_pipeline_version(
            pipeline_bundle_path,
            name=pipeline_version_name,
            pipelineid=pipeline_id,
        )

        logger.info(
            f"Uploaded version {pipeline_version.name} to pipeline id {pipeline_id}. "
            f"{self.kfp_host}/#/pipelines/details/{pipeline_id}"
        )

        return f"{self.kfp_host}/#/pipelines/details/{pipeline_id}"

    def get_latest_experiment_from_name(
        self, experiment_name: str
    ) -> Optional[ApiListExperimentsResponse]:
        """
        Retrieves the latest experiment with a given name.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            Optional[ApiListExperimentsResponse]: The latest experiment with the given name, or None if not found.
        """
        filter = json.dumps(
            {
                "predicates": [
                    {
                        "key": "name",
                        "op": _KfpApiFilterOperations.EQUALS.value,
                        "string_value": experiment_name,
                    }
                ]
            }
        )
        resp_api_list: ApiListExperimentsResponse = self.kfp_client.list_experiments(
            sort_by="created_at desc", filter=filter
        )
        if len(resp_api_list.experiments) > 0:
            return resp_api_list.experiments[0]
        return None

    def get_latest_run_with_name(
        self,
        kfp_run_name: str,
        experiment_name: str,
    ) -> Optional[ApiRunDetailWrapper]:
        """
        Retrieves the latest run with a given name and experiment name.

        Args:
            kfp_run_name (str): The name of the run.
            experiment_name (str): The name of the experiment.

        Returns:
            Optional[ApiRunDetailWrapper]: The latest run with the given name and experiment name, or None if not found.
        """
        experiment: ApiListExperimentsResponse = self.get_latest_experiment_from_name(
            experiment_name=experiment_name
        )
        experiment_id = experiment.id if experiment else None
        # filter tries to find runs with name == kfp_run_name
        filter = json.dumps(
            {
                "predicates": [
                    {
                        "key": "name",
                        "op": _KfpApiFilterOperations.EQUALS.value,
                        "string_value": kfp_run_name,
                    }
                ]
            }
        )

        resp_api_list: ApiListRunsResponse = self.kfp_client.list_runs(
            experiment_id=experiment_id, sort_by="created_at desc", filter=filter
        )
        if len(resp_api_list.runs) > 0:
            api_run: ApiRun = resp_api_list.runs[0]
            return self.get_run(run_id=api_run.id)
        return None

    def get_run(
        self,
        run_id: str,
    ) -> ApiRunDetailWrapper:
        """
        Retrieves the details of a run given its ID.

        Args:
            run_id (str): The ID of the run.

        Returns:
            ApiRunDetailWrapper: The details of the run.
        """
        resp_api_run: ApiRunDetail = self.kfp_client.get_run(
            run_id=run_id,
        )
        return ApiRunDetailWrapper(api_run=resp_api_run)

    def wait_for_run_completion(self, run_id: str, timeout: float = 7200) -> None:
        """
        Waits for a run to complete.

        Args:
            run_id (str): The ID of the run.
            timeout (float): The maximum time to wait for the run to complete, in seconds. Defaults to 7200.

        Returns:
            None
        """
        try:
            run_response = self.kfp_client.wait_for_run_completion(
                run_id=run_id, timeout=timeout
            )
            if run_response.run.status == "Succeeded":
                logger.info("KFP finished with status Succeeded!")
                return run_response.run.status
            else:
                raise RuntimeError(
                    f"KFP run stop with status: {run_response.run.status}. "
                    f"Please check the KFP page to trace down the error @ {self.kfp_host}/#/runs/details/{run_id}"
                )
        except Exception as e:
            logger.error(
                f"Error when waiting on KFP run {self.kfp_host}/#/runs/details/{run_id} to finish:"
            )
            raise e

    @lru_cache(maxsize=1)
    def get_host_k8_cluster_name(
        self,
    ) -> str:
        """
        Retrieves the name of the Kubernetes cluster that the KFP instance is running on.

        Returns:
            str: The name of the Kubernetes cluster.
        """
        request_url = f"{self.kfp_host}/system/cluster-name"
        response = requests.get(request_url, headers=self.__get_auth_header())
        return response.text

    @lru_cache(maxsize=1)
    def get_host_gcp_project_name(
        self,
    ) -> str:
        """
        Retrieves the name of the GCP project that the KFP instance is running on.

        Returns:
            str: The name of the GCP project.
        """
        request_url = f"{self.kfp_host}/system/project-id"
        response = requests.get(request_url, headers=self.__get_auth_header())
        return response.text
