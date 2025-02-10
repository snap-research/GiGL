from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from gigl.common.logger import Logger
from gigl.common.utils.func_tools import lru_cache
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    NodeType,
    Relation,
)
from snapchat.research.gbml import graph_schema_pb2

logger = Logger()


@dataclass
class GraphMetadataPbWrapper:
    graph_metadata_pb: graph_schema_pb2.GraphMetadata

    __condensed_edge_type_to_condensed_node_types: Dict[
        CondensedEdgeType, Tuple[CondensedNodeType, CondensedNodeType]
    ] = field(init=False)
    __hash: int = field(init=False)

    def __post_init__(self):
        # Check that the graph metadata contains condensed node and edge types.
        if not (
            self.graph_metadata_pb.condensed_node_type_map
            and self.graph_metadata_pb.condensed_edge_type_map
        ):
            raise ValueError(
                "Graph metadata does not contain condensed node and edge type metadata. "
                "Please use ConfigPopulator to populate the graph metadata or designate it yourself."
            )

        # Populate the __condensed_edge_type_to_condensed_node_types field.
        node_type_to_condensed_node_types: Dict[NodeType, CondensedNodeType] = dict()
        for (
            condensed_node_type,
            node_type,
        ) in self.graph_metadata_pb.condensed_node_type_map.items():
            node_type_to_condensed_node_types[NodeType(node_type)] = CondensedNodeType(
                condensed_node_type
            )

        condensed_edge_type_to_condensed_node_types: Dict[
            CondensedEdgeType, Tuple[CondensedNodeType, CondensedNodeType]
        ] = dict()
        for condensed_edge_type in self.graph_metadata_pb.condensed_edge_type_map:
            edge_type_pb = self.graph_metadata_pb.condensed_edge_type_map[
                condensed_edge_type
            ]
            src_condensed_node_type = CondensedNodeType(
                node_type_to_condensed_node_types[NodeType(edge_type_pb.src_node_type)]
            )
            dst_condensed_node_type = CondensedNodeType(
                node_type_to_condensed_node_types[NodeType(edge_type_pb.dst_node_type)]
            )
            condensed_edge_type_to_condensed_node_types[
                CondensedEdgeType(condensed_edge_type)
            ] = (src_condensed_node_type, dst_condensed_node_type)

        self.__condensed_edge_type_to_condensed_node_types = (
            condensed_edge_type_to_condensed_node_types
        )

        self.__hash = hash(
            (
                tuple(sorted(self.graph_metadata_pb.condensed_edge_type_map.keys())),
                tuple(
                    (
                        edge_type_pb.src_node_type,
                        edge_type_pb.relation,
                        edge_type_pb.dst_node_type,
                    )
                    for _, edge_type_pb in sorted(
                        self.graph_metadata_pb.condensed_edge_type_map.items()
                    )
                ),
                tuple(sorted(self.graph_metadata_pb.condensed_node_type_map.keys())),
                tuple(sorted(self.graph_metadata_pb.condensed_node_type_map.values())),
            )
        )

    @property
    def condensed_edge_type_to_condensed_node_types(
        self,
    ) -> Dict[CondensedEdgeType, Tuple[CondensedNodeType, CondensedNodeType]]:
        """
        Allows access to a mapping which simplifies looking up src/dst
        CondensedNodeTypes for each CondensedEdgeType.
        :return:
        """

        return self.__condensed_edge_type_to_condensed_node_types

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def condensed_node_type_to_node_type_map(self) -> Dict[CondensedNodeType, NodeType]:
        return {
            CondensedNodeType(condensed_node_type): NodeType(node_type)
            for condensed_node_type, node_type in self.graph_metadata_pb.condensed_node_type_map.items()
        }

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def node_type_to_condensed_node_type_map(self) -> Dict[NodeType, CondensedNodeType]:
        return {v: k for k, v in self.condensed_node_type_to_node_type_map.items()}

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def condensed_edge_type_to_edge_type_map(self) -> Dict[CondensedEdgeType, EdgeType]:
        return {
            CondensedEdgeType(condensed_edge_type): EdgeType(
                src_node_type=NodeType(edge_type.src_node_type),
                relation=Relation(edge_type.relation),
                dst_node_type=NodeType(edge_type.dst_node_type),
            )
            for condensed_edge_type, edge_type in self.graph_metadata_pb.condensed_edge_type_map.items()
        }

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def edge_type_to_condensed_edge_type_map(self) -> Dict[EdgeType, CondensedEdgeType]:
        return {v: k for k, v in self.condensed_edge_type_to_edge_type_map.items()}

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def edge_types(self) -> List[EdgeType]:
        return list(self.condensed_edge_type_to_edge_type_map.values())

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def node_types(self) -> List[NodeType]:
        return list(self.condensed_node_type_to_node_type_map.values())

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def condensed_edge_types(self) -> List[CondensedEdgeType]:
        return list(self.condensed_edge_type_to_edge_type_map.keys())

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def condensed_node_types(self) -> List[CondensedNodeType]:
        return list(self.condensed_node_type_to_node_type_map.keys())

    @property  # type: ignore
    @lru_cache(maxsize=1)
    def is_heterogeneous(self) -> bool:
        return len(self.edge_types) > 1 or len(self.node_types) > 1

    def __hash__(self) -> int:
        return self.__hash
