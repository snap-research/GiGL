from collections import abc
from dataclasses import dataclass
from typing import Optional, Union

import torch

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.distributed import (
    NEGATIVE_LABEL_RELATION,
    POSITIVE_LABEL_RELATION,
    to_heterogeneous_edge,
    to_heterogeneous_node,
)

logger = Logger()


# This dataclass should not be frozen, as we are expected to delete its members once they have been registered inside of the partitioner
# in order to save memory.
@dataclass
class LoadedGraphTensors:
    # Unpartitioned Node Ids
    node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    # Unpartitioned Node Features
    node_features: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]
    # Unpartitioned Edge Index
    edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    # Unpartitioned Edge Features
    edge_features: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Positive Edge Label
    positive_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    # Unpartitioned Negative Edge Label
    negative_label: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]

    def treat_labels_as_edges(self) -> None:
        """Convert positive and negative labels to edges. Converts this object in-place to a "heterogeneous" representation.

        This requires the following conditions and will throw if they are not met:
            1. The node_ids, node_features, edge_index, and edge_features are not dictionaries (we loaded a homogeneous graph).
            2. The positive_label and negative_label are not None and are Tensors, not dictionaries.
        """
        # TODO(kmonte): We should support heterogeneous graphs in the future.
        if (
            isinstance(self.node_ids, abc.Mapping)
            or isinstance(self.node_features, abc.Mapping)
            or isinstance(self.edge_index, abc.Mapping)
            or isinstance(self.edge_features, abc.Mapping)
            or isinstance(self.positive_label, abc.Mapping)
            or isinstance(self.negative_label, abc.Mapping)
        ):
            raise ValueError(
                "Cannot treat labels as edges when using heterogeneous graph tensors."
            )
        if self.positive_label is None or self.negative_label is None:
            raise ValueError(
                "Cannot treat labels as edges when positive or negative labels are None."
            )

        edge_index_with_labels = to_heterogeneous_edge(self.edge_index)
        main_edge_type = next(iter(edge_index_with_labels.keys()))
        logger.info(
            f"Basing positive and negative labels on edge types on main label edge type: {main_edge_type}."
        )
        positive_label_edge_type = EdgeType(
            main_edge_type.src_node_type,
            POSITIVE_LABEL_RELATION,
            main_edge_type.dst_node_type,
        )
        edge_index_with_labels[positive_label_edge_type] = self.positive_label
        negative_label_edge_type = EdgeType(
            main_edge_type.src_node_type,
            NEGATIVE_LABEL_RELATION,
            main_edge_type.dst_node_type,
        )
        edge_index_with_labels[negative_label_edge_type] = self.negative_label
        logger.info(
            f"Treating positive labels as edge type {positive_label_edge_type} and negative labels as edge type {negative_label_edge_type}."
        )

        self.node_ids = to_heterogeneous_node(self.node_ids)
        self.node_features = to_heterogeneous_node(self.node_features)
        self.edge_index = edge_index_with_labels
        self.edge_features = to_heterogeneous_edge(self.edge_features)
        self.positive_label = None
        self.negative_label = None
