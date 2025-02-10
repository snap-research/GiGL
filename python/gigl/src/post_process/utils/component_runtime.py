from typing import Dict, Optional

from gigl.common.services.kfp import KFPService
from gigl.common.types.wrappers.argo_workflow_manifest import (
    ArgoWorkflowManifestWrapper,
)
from gigl.common.types.wrappers.kfp_api import ApiRunDetailWrapper


def get_runtime_manifest_from_kfp_pipeline(
    kfp_service: KFPService, experiment_name: str, kfp_run_name: str
) -> ArgoWorkflowManifestWrapper:
    pipeline_run_detail: Optional[ApiRunDetailWrapper] = (
        kfp_service.get_latest_run_with_name(
            kfp_run_name=kfp_run_name, experiment_name=experiment_name
        )
    )
    assert pipeline_run_detail is not None
    manifest = pipeline_run_detail.workflow_manifest
    return manifest


def assert_component_runtimes_match_expected_parameters(
    runtime_manifest: ArgoWorkflowManifestWrapper,
    component_name_runtime_hr: Dict[str, int],
) -> None:
    checked_components = set()
    for component_name, expected_runtime_hr in component_name_runtime_hr.items():
        for (
            _,
            pipeline_node_data,
        ) in runtime_manifest.component_status_by_component_display_name.items():
            if pipeline_node_data.display_name == component_name:
                t_start = pipeline_node_data.started_at
                t_finish = pipeline_node_data.finished_at
                runtime_sec = (t_finish - t_start).seconds
                expected_runtime_sec = expected_runtime_hr * 3600
                if runtime_sec > expected_runtime_sec:
                    raise ValueError(
                        f"Component {component_name} took longer than expected runtime of {expected_runtime_hr} hrs. Actual runtime was {t_finish- t_start}."
                    )
                else:
                    checked_components.add(component_name)
    if len(checked_components) != len(component_name_runtime_hr):
        raise ValueError(
            f"run time check completed only for {checked_components}; components {component_name_runtime_hr.keys()- checked_components} not found in pipeline runtime manifest."
        )
