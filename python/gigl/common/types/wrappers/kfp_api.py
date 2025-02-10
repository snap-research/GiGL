import datetime
from typing import Dict

from kfp_server_api.models.api_parameter import ApiParameter
from kfp_server_api.models.api_pipeline_runtime import ApiPipelineRuntime
from kfp_server_api.models.api_pipeline_spec import ApiPipelineSpec
from kfp_server_api.models.api_run_detail import ApiRunDetail

from gigl.common.types.wrappers.argo_workflow_manifest import (
    ArgoWorkflowManifestWrapper,
)
from gigl.common.utils.func_tools import lru_cache


class ApiRunDetailWrapper:
    def __init__(self, api_run: ApiRunDetail) -> None:
        self._api_run = api_run

    @property
    def api_run(self) -> ApiRunDetail:
        return self._api_run

    @property
    def created_at(self) -> datetime.datetime:
        return self.api_run.run.created_at

    @property
    def finished_at(self) -> datetime.datetime:
        return self.api_run.run.finished_at

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def job_parameters(self) -> Dict[str, str]:
        parameters_dict: Dict[str, str] = {}
        pipeline_spec: ApiPipelineSpec = self.api_run.run.pipeline_spec
        param: ApiParameter
        for param in pipeline_spec.parameters:
            parameters_dict[param.name] = param.value

        return parameters_dict

    @property
    def workflow_manifest(self) -> ArgoWorkflowManifestWrapper:
        pipeline_runtime: ApiPipelineRuntime = self._api_run.pipeline_runtime
        manifest = ArgoWorkflowManifestWrapper(pipeline_runtime.workflow_manifest)
        return manifest
