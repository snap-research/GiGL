from dataclasses import dataclass
from typing import Dict, Optional, Union

from gigl.common.data.dataloaders import SerializedTFRecordInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType


@dataclass(frozen=True)
class DatasetInputMetadata:
    """
    Stores information for all entities. If homogeneous, all types are of type SerializedTFRecordInfo. Otherwise, they are dictionaries with the corresponding mapping.
    These fields are used to store inputs to GLT's DistDataset.load() function. This is done separate from existing GiGL constructs such as PreprocessedMetadataPbWrapper so that
    there is not a strict coupling between GiGL orchestration and the GLT Dataset layer.
    """

    # Node Entity Info for loading node tensors, a SerializedTFRecordInfo for homogeneous and Dict[NodeType, SerializedTFRecordInfo] for heterogeneous cases
    node_entity_info: Union[
        SerializedTFRecordInfo, Dict[NodeType, SerializedTFRecordInfo]
    ]
    # Edge Entity Info for loading edge tensors, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases
    edge_entity_info: Union[
        SerializedTFRecordInfo, Dict[EdgeType, SerializedTFRecordInfo]
    ]
    # Positive Label Entity Info, if present, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases. May be None
    # for specific edge types. If data has no positive labels across all edge types, this value is None
    positive_label_entity_info: Optional[
        Union[SerializedTFRecordInfo, Dict[EdgeType, Optional[SerializedTFRecordInfo]]]
    ] = None
    # Negative Label Entity Info, if present, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases. May be None
    # for specific edge types. If input has no negative labels across all edge types, this value is None.
    negative_label_entity_info: Optional[
        Union[SerializedTFRecordInfo, Dict[EdgeType, Optional[SerializedTFRecordInfo]]]
    ] = None


# TODO (mkolodner-sc): Remove below comment which showes example usage of DatasetInputMetadata when the DistLinkPredictionDataset class is ready
"""
Example usage of DatasetInputMetadata class for a GiGL-orchestrated job

class DistLinkPredictionDataset():

    ...

    def load(dataset_input_metadata: DatasetInputMetadata):
        tfrecord_dataloader = TFRecordDataloader(rank=0, world_size=1)

        ...
        
        # Heterogeneous Example
        for node_type in dataset_input_metadata.node_entity_info:
            node_ids[node_type], node_features[node_type] = tfrecord_dataloader.load_as_torch_tensors(serialized_tf_record_info=dataset_input_metadata.node_entity_info[node_type])

            ...

dataset = DistLinkPredictionDataset(...)

# Generating DatasetInputMetadata

dataset_input_metadata = convert_pb_to_dataset_input_metadata(preprocessed_metadata_pb_wrapper, graph_metadata_pb_wrapper)

# Passing DatasetInputMetadata into DistLinkPredictionDataset class, which loads, partitions, and stores all of the relevant information on the current rank

dataset.load(dataset_input_metadata)

# DistLinkPredictionDataset instance will be eventually passed into GLT's DistNeighborLoader in the training/inference loop for live subgraph sampling.
"""
