from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import torch

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

DEFAULT_HOMOGENEOUS_NODE_TYPE = NodeType("default_homogeneous_node_type")
DEFAULT_HOMOGENEOUS_EDGE_TYPE = EdgeType(
    src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
    relation=Relation("to"),
    dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
)


class EdgeAssignStrategy(Enum):
    BY_SOURCE_NODE = "BY_SOURCE_NODE"
    BY_DESTINATION_NODE = "BY_DESTINATION_NODE"


@dataclass(frozen=True)
class FeaturePartitionData:
    """Data and indexing info of a node/edge feature partition."""

    # node/edge feature tensor
    feats: torch.Tensor
    # node/edge ids tensor corresponding to `feats`
    ids: torch.Tensor


@dataclass(frozen=True)
class GraphPartitionData:
    """Data and indexing info of a graph partition."""

    # edge index (rows, cols)
    edge_index: torch.Tensor
    # edge ids tensor corresponding to `edge_index`
    edge_ids: torch.Tensor
    # weights tensor corresponding to `edge_index`
    weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class PartitionOutput:
    # Node partition book
    node_partition_book: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]

    # Edge partition book
    edge_partition_book: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]

    # Partitioned edge index on current rank
    partitioned_edge_index: Union[
        GraphPartitionData, Dict[EdgeType, GraphPartitionData]
    ]

    # Node features on current rank, May be None if node features are not partitioned
    partitioned_node_features: Optional[
        Union[FeaturePartitionData, Dict[NodeType, FeaturePartitionData]]
    ]

    # Edge features on current rank, May be None if edge features are not partitioned
    partitioned_edge_features: Optional[
        Union[FeaturePartitionData, Dict[EdgeType, FeaturePartitionData]]
    ]

    # Positive edge indices on current rank, May be None if positive edge labels are not partitioned
    partitioned_positive_labels: Optional[
        Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ]

    # Negative edge indices on current rank, May be None if negative edge labels are not partitioned
    partitioned_negative_labels: Optional[
        Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ]
