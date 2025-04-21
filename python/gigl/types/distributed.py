from dataclasses import dataclass
from typing import Optional, TypeVar, Union, overload

import torch
from graphlearn_torch.partition import PartitionBook

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

DEFAULT_HOMOGENEOUS_NODE_TYPE = NodeType("default_homogeneous_node_type")
DEFAULT_HOMOGENEOUS_EDGE_TYPE = EdgeType(
    src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
    relation=Relation("to"),
    dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

POSITIVE_LABEL_RELATION = Relation("positive_label")
NEGATIVE_LABEL_RELATION = Relation("negative_label")


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


# This dataclass should not be frozen, as we are expected to delete partition outputs once they have been registered inside of GLT DistDataset
# in order to save memory.
@dataclass
class PartitionOutput:
    # Node partition book
    node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]]

    # Edge partition book
    edge_partition_book: Union[PartitionBook, dict[EdgeType, PartitionBook]]

    # Partitioned edge index on current rank. This field will always be populated after partitioning. However, we may set this
    # field to None during dataset.build() in order to minimize the peak memory usage, and as a result type this as Optional.
    partitioned_edge_index: Optional[
        Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]
    ]

    # Node features on current rank, May be None if node features are not partitioned
    partitioned_node_features: Optional[
        Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
    ]

    # Edge features on current rank, May be None if edge features are not partitioned
    partitioned_edge_features: Optional[
        Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]
    ]

    # Positive edge indices on current rank, May be None if positive edge labels are not partitioned
    partitioned_positive_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]

    # Negative edge indices on current rank, May be None if negative edge labels are not partitioned
    partitioned_negative_labels: Optional[
        Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ]


_T = TypeVar("_T")


@overload
def to_heterogeneous_node(x: None) -> None:
    ...


@overload
def to_heterogeneous_node(x: Union[_T, dict[NodeType, _T]]) -> dict[NodeType, _T]:
    ...


def to_heterogeneous_node(
    x: Optional[Union[_T, dict[NodeType, _T]]]
) -> Optional[dict[NodeType, _T]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return {DEFAULT_HOMOGENEOUS_NODE_TYPE: x}


@overload
def to_heterogeneous_edge(x: None) -> None:
    ...


@overload
def to_heterogeneous_edge(x: Union[_T, dict[EdgeType, _T]]) -> dict[EdgeType, _T]:
    ...


def to_heterogeneous_edge(
    x: Optional[Union[_T, dict[EdgeType, _T]]]
) -> Optional[dict[EdgeType, _T]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return {DEFAULT_HOMOGENEOUS_EDGE_TYPE: x}


@overload
def to_homogeneous(x: None) -> None:
    ...


@overload
def to_homogeneous(x: Union[_T, dict[Union[NodeType, EdgeType], _T]]) -> _T:
    ...


def to_homogeneous(
    x: Optional[Union[_T, dict[Union[NodeType, EdgeType], _T]]]
) -> Optional[_T]:
    if x is None:
        return None
    if isinstance(x, dict):
        if len(x) != 1:
            raise ValueError(
                f"Expected a single value in the dictionary, but got multiple keys: {x.keys()}"
            )
        n = next(iter(x.values()))
        return n
    return x
