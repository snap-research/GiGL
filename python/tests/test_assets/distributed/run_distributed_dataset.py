from typing import Literal, MutableMapping, Optional

from gigl.common.data.load_torch_tensors import TFDatasetOptions
from gigl.common.utils.vertex_ai_context import DistributedContext
from gigl.distributed.dataset_factory import _build_dataset_process, build_dataset
from gigl.distributed.dist_link_prediction_data_partitioner import (
    DistLinkPredictionDataPartitioner,
)
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.utils.data_splitters import NodeAnchorLinkSplitter


def run_distributed_dataset(
    rank: int,
    world_size: int,
    mocked_dataset_info: MockedDatasetInfo,
    output_dict: MutableMapping[int, DistLinkPredictionDataset],
    should_load_tensors_in_parallel: bool,
    master_ip_address: str,
    master_port: int,
    partitioner: Optional[DistLinkPredictionDataPartitioner] = None,
    dataset: Optional[DistLinkPredictionDataset] = None,
    splitter: Optional[NodeAnchorLinkSplitter] = None,
) -> DistLinkPredictionDataset:
    """
    Runs DistLinkPredictionDataset Load() __init__ and load() functions provided a mocked dataset info
    Args:
        rank (int): Rank of the current process
        world_size (int): World size of the current process
        mocked_dataset_info (MockedDatasetInfo): Mocked Dataset Metadata for current run
        output_dict (MutableMapping[int, DistLinkPredictionDataset]): Dict initialized by mp.Manager().dict() in which outputs will be written to
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        master_ip_address (str): Master IP Address for performing distributed operations.
        master_port (int) Master Port for performing distributed operations
        partitioner (Optional[DistLinkPredictionDataPartitioner]): Optional initialized partitioner class to pass into `load_and_build_partitioned_dataset`
        dataset (Optional[DistLinkPredictionDataset]): Optional initialized dataset class to pass into `load_and_build_partitioned_dataset`
    """
    mocked_dataset_artifact_metadata: MockedDatasetArtifactMetadata = (
        get_mocked_dataset_artifact_metadata()[mocked_dataset_info.name]
    )
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=mocked_dataset_artifact_metadata.frozen_gbml_config_uri
    )
    preprocessed_metadata_pb_wrapper = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
    )
    graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
    )

    distributed_context = DistributedContext(
        main_worker_ip_address=master_ip_address,
        global_rank=rank,
        global_world_size=world_size,
    )

    sample_edge_direction: Literal["in", "out"] = "out"

    if partitioner is None and dataset is None:
        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=distributed_context,
            sample_edge_direction=sample_edge_direction,
            should_load_tensors_in_parallel=should_load_tensors_in_parallel,
            partitioner=partitioner,
            dataset=dataset,
            splitter=splitter,
        )
        output_dict[rank] = dataset
        return dataset
    else:
        # In testing, we pass in a Partitioner or Dataset class with a NotImplementedError in order to ensure
        # that custom logic from children partitioners and datasets are showcased in the DatasetFactory. As a result,
        # we must use `_dataset_building_process` instead of `launch_dataset_building_process` so that a spawned process
        # is not launched, which would cause raise expected errors immediately instead of being caught by the unit test.
        _build_dataset_process(
            process_number_on_current_machine=0,
            output_dict={},
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=distributed_context,
            dataset_building_port=master_port,
            sample_edge_direction=sample_edge_direction,
            should_load_tensors_in_parallel=should_load_tensors_in_parallel,
            partitioner=partitioner,
            dataset=dataset,
            tf_dataset_options=TFDatasetOptions(),
        )
        assert dataset is not None
        return dataset
