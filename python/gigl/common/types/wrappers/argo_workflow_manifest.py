import json
from datetime import datetime
from typing import Dict

import argo_workflows.models as models
from argo_workflows.model.io_argoproj_workflow_v1alpha1_node_status import (
    IoArgoprojWorkflowV1alpha1NodeStatus,
)
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow_status import (
    IoArgoprojWorkflowV1alpha1WorkflowStatus,
)

from gigl.common.utils.func_tools import lru_cache

ARGO_WORFLOW_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class ArgoWorkflowNodeStatus:
    """Class allows us to create a strictly types wrapper around IoArgoprojWorkflowV1alpha1NodeStatus
    to reduce cognitive load when working with the Argo Workflow API.
    """

    def __init__(
        self,
        io_argoproj_workflow_v1alpha1_node_status: IoArgoprojWorkflowV1alpha1NodeStatus,
    ) -> None:
        self._io_argoproj_workflow_v1alpha1_node_status: (
            IoArgoprojWorkflowV1alpha1NodeStatus
        ) = io_argoproj_workflow_v1alpha1_node_status

    @property
    def pod_name(self) -> str:
        return self._io_argoproj_workflow_v1alpha1_node_status.id

    @property
    def display_name(self) -> str:
        return self._io_argoproj_workflow_v1alpha1_node_status.displayName

    @property
    def finished_at(self) -> datetime:
        return datetime.strptime(
            self._io_argoproj_workflow_v1alpha1_node_status.finishedAt,
            ARGO_WORFLOW_DATETIME_FORMAT,
        )

    @property
    def started_at(self) -> datetime:
        return datetime.strptime(
            self._io_argoproj_workflow_v1alpha1_node_status.startedAt,
            ARGO_WORFLOW_DATETIME_FORMAT,
        )


class ArgoWorkflowManifestWrapper:
    def __init__(self, workflow_manifest_json_str: str) -> None:
        manifest_dict = json.loads(workflow_manifest_json_str)
        self.argo_workflow: models.IoArgoprojWorkflowV1alpha1Workflow = (
            models.IoArgoprojWorkflowV1alpha1Workflow(
                **manifest_dict, _check_type=False
            )
        )

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def __node_status(self) -> Dict[str, IoArgoprojWorkflowV1alpha1NodeStatus]:
        status = self.argo_workflow.status
        workflow_status = IoArgoprojWorkflowV1alpha1WorkflowStatus(
            **status, _check_type=False
        )
        return {
            k8_node_name: IoArgoprojWorkflowV1alpha1NodeStatus(
                **pipeline_node_data, _check_type=False
            )
            for k8_node_name, pipeline_node_data in workflow_status.nodes.items()
        }

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def component_status_by_component_display_name(
        self,
    ) -> Dict[str, ArgoWorkflowNodeStatus]:
        """

        Returns:
            Dict[str, ArgoWorkflowNodeStatus]: Note that the keys are the display name of the KFP component
        """
        return {
            node_status.displayName: ArgoWorkflowNodeStatus(node_status)
            for node_status in self.__node_status.values()
        }
