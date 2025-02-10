from typing import NamedTuple


def log_eval_metrics_to_ui(
    task_config_uri: str,
    component: str,
) -> NamedTuple(  # type: ignore
    "Outputs",
    [
        ("mlpipeline_metrics", "Metrics"),
    ],
):
    """Returns model evaluation metrics produced by trainer, such
    that they are parsable by the Kubeflow Pipelines UI.
    Args:
        task_config_uri (str,): _description_
        component (str,): _description_
    Returns:
        _type_: _description_
    """

    # This is required to resolve below packages when containerized by KFP.
    import os
    import sys

    from google.api_core.exceptions import NotFound

    sys.path.append(os.getcwd())

    from gigl.common import UriFactory
    from gigl.common.logger import Logger
    from gigl.common.utils.proto_utils import ProtoUtils
    from gigl.src.common.constants.components import GiGLComponents
    from gigl.src.common.utils.file_loader import FileLoader
    from snapchat.research.gbml import gbml_config_pb2

    logger = Logger()
    proto_utils = ProtoUtils()
    gbml_config_pb = proto_utils.read_proto_from_yaml(
        uri=UriFactory.create_uri(task_config_uri), proto_cls=gbml_config_pb2.GbmlConfig
    )
    logger.info(f"Read gbml config pb from {task_config_uri}")
    if component == GiGLComponents.Trainer.value:
        eval_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.trained_model_metadata.eval_metrics_uri
        )
    elif component == GiGLComponents.PostProcessor.value:
        eval_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.postprocessed_metadata.post_processor_log_metrics_uri
        )
    else:
        raise ValueError(f"Unknown component: {component}")

    logger.info(f"Fetching eval metrics from: {eval_metrics_uri.uri}")

    file_loader = FileLoader()
    metrics_str: str

    try:
        tfh = file_loader.load_to_temp_file(file_uri_src=eval_metrics_uri)
        with open(tfh.name, "r") as f:
            metrics_str = f.read()
    except (NotFound, FileNotFoundError) as e:
        logger.warning(
            f"Error loading metrics file: {e}, evaluation could have been skipped"
        )
        return [{"metrics": []}]

    logger.info(f"Got metrics_str: {metrics_str}")

    return [metrics_str]
