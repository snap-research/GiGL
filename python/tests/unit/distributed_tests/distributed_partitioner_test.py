# Originally taken from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/test/python/test_dist_random_partitioner.py

import unittest
from collections import abc, defaultdict
from typing import Dict, Iterable, List, MutableMapping, Tuple, Union

import graphlearn_torch as glt
import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import init_rpc, init_worker_group
from parameterized import param, parameterized
from torch.multiprocessing import Manager

from gigl.distributed.partitioner.dist_link_prediction_data_partitioner import (
    DistLinkPredictionDataPartitioner,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.distributed import (
    EdgeAssignStrategy,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from tests.test_assets.distributed.constants import (
    EDGE_TYPE_TO_FEATURE_DIMENSION_MAP,
    ITEM_NODE_TYPE,
    MOCKED_HETEROGENEOUS_EDGE_TYPES,
    MOCKED_HETEROGENEOUS_NODE_TYPES,
    MOCKED_NUM_PARTITIONS,
    MOCKED_UNIFIED_GRAPH,
    NODE_TYPE_TO_FEATURE_DIMENSION_MAP,
    RANK_TO_MOCKED_GRAPH,
    RANK_TO_NODE_TYPE_TYPE_TO_NUM_NODES,
    USER_NODE_TYPE,
    USER_TO_USER_EDGE_TYPE,
)
from tests.test_assets.distributed.run_distributed_partitioner import (
    InputDataStrategy,
    run_distributed_partitioner,
)
from tests.test_assets.distributed.utils import assert_tensor_equality


class DistRandomPartitionerTestCase(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"

    def _assert_data_type_correctness(
        self,
        output_data: Union[
            torch.Tensor,
            Dict[NodeType, torch.Tensor],
            Dict[EdgeType, torch.Tensor],
            FeaturePartitionData,
            Dict[NodeType, FeaturePartitionData],
            Dict[EdgeType, FeaturePartitionData],
            GraphPartitionData,
            Dict[EdgeType, GraphPartitionData],
        ],
        is_heterogeneous: bool,
        expected_entity_types: Union[List[EdgeType], List[NodeType]],
    ):
        """
        Checks that each item in the provided output data is correctly typed and, if heterogeneous, that edge types and node types are as expected.
        Args:
            output_data(Union[
                torch.Tensor,
                Dict[NodeType, torch.Tensor],
                Dict[EdgeType, torch.Tensor],
                FeaturePartitionData,
                Dict[NodeType, FeaturePartitionData],
                Dict[EdgeType, FeaturePartitionData],
                GraphPartitionData,
                Dict[EdgeType, GraphPartitionData],
            ]): Items of which correctness is being checked for
            is_heterogeneous (bool): Whether the provided input data is heterogeneous or homogeneous
            expected_entity_types(Union[List[EdgeType], List[NodeType]]): Expected node or edge type which we are checking against for heterogeneous inputs
        """
        self.assertIsNotNone(output_data)
        if is_heterogeneous:
            assert isinstance(
                output_data, abc.Mapping
            ), "Homogeneous output detected for heterogeneous input"
            self.assertTrue(sorted(output_data.keys()) == sorted(expected_entity_types))
        else:
            self.assertNotIsInstance(output_data, abc.Mapping)

    def _assert_graph_outputs(
        self,
        rank: int,
        is_heterogeneous: bool,
        edge_assign_strategy: EdgeAssignStrategy,
        output_node_partition_book: Union[torch.Tensor, Dict[NodeType, torch.Tensor]],
        output_edge_partition_book: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
        output_edge_index: Union[
            GraphPartitionData, Dict[EdgeType, GraphPartitionData]
        ],
        expected_node_types: List[NodeType],
        expected_edge_types: List[EdgeType],
    ) -> None:
        """
        Checks correctness for graph outputs of partitioning, including node partition book, edge partition book, and the graph edge indices + ids
        Args:
            rank (int): Rank from current output
            is_heterogeneous (bool): Whether the output is expected to be homogeneous or heterogeneous
            edge_assign_strategy (EdgeAssignStrategy): Whether to partion edges according to the partition book of the source node or destination node
            output_node_partition_book (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): Node Partition Book from partitioning, either a Tensor if homogeneous or a Dict[NodeType, Tensor] if heterogeneous
            output_edge_partition_book (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Edge Partition Book from partitioning, either a Tensor if homogeneous or a Dict[EdgeType, Tensor] if heterogeneous
            output_edge_index: (Union[GraphPartitionData, Dict[EdgeType, GraphPartitionData]]): Output edge indices and ids from partitioning, either a GraphPartitionData if homogeneous or a Dict[EdgeType, GraphPartitionData] if heterogeneous
            expected_node_types (List[NodeType]): Expected node types for heterogeneous input
            expected_edge_types (List[EdgeType]): Expected edge types for heterogeneous input
        """
        self._assert_data_type_correctness(
            output_data=output_node_partition_book,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_node_types,
        )
        self._assert_data_type_correctness(
            output_data=output_edge_partition_book,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )
        self._assert_data_type_correctness(
            output_data=output_edge_index,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )
        # To unify logic between homogeneous and heterogeneous cases, we define an iterable which we'll loop over.
        # Each iteration contains an EdgeType, an edge partition book, and a graph consisting of edge indices and ids.
        entity_iterable: Iterable[Tuple[EdgeType, torch.Tensor, GraphPartitionData]]
        if isinstance(output_edge_partition_book, abc.Mapping) and isinstance(
            output_edge_index, abc.Mapping
        ):
            entity_iterable = [
                (
                    edge_type,
                    output_edge_partition_book[edge_type],
                    output_edge_index[edge_type],
                )
                for edge_type in MOCKED_HETEROGENEOUS_EDGE_TYPES
            ]

        elif isinstance(output_edge_partition_book, torch.Tensor) and isinstance(
            output_edge_index, GraphPartitionData
        ):
            entity_iterable = [
                (USER_TO_USER_EDGE_TYPE, output_edge_partition_book, output_edge_index)
            ]
        else:
            raise ValueError(
                f"The output edge partition book of type {type(output_edge_partition_book)} and the output graph of type {type(output_edge_index)} are not compatible."
            )

        for edge_type, edge_partition_book, graph in entity_iterable:
            node_partition_book: torch.Tensor
            node_ids: torch.Tensor

            self.assertEqual(graph.edge_index.size(0), 2)
            # We take the unique items in either source or destination nodes, as source/destination node ids which
            # repeat across multiple edges will still only take up one slot in the partition book.
            if edge_assign_strategy == EdgeAssignStrategy.BY_SOURCE_NODE:
                target_node_type = edge_type.src_node_type
                node_ids = torch.unique(graph.edge_index[0])
            else:
                target_node_type = edge_type.dst_node_type
                node_ids = torch.unique(graph.edge_index[1])

            num_nodes_on_rank: int = RANK_TO_NODE_TYPE_TYPE_TO_NUM_NODES[rank][
                target_node_type
            ]

            if isinstance(output_node_partition_book, abc.Mapping):
                node_partition_book = output_node_partition_book[target_node_type]
            else:
                node_partition_book = output_node_partition_book

            # We expect the number of output node ids to be the same as the number of nodes input to the partitioner on the current rank, as the inputs and outputs per rank
            # should both be equal in number across all ranks. This is because each node id is the target node of exactly one edge in the mocked graph.
            self.assertEqual(node_ids.size(0), num_nodes_on_rank)

            # We expect the target node ids on the current rank to be correctly specified in the node partition book. For example, if we are on rank 0 with
            # target node ids [1, 3, 5], we expected node_partition_book[[1, 3, 5]] = 0.
            expected_node_pidx = torch.ones(num_nodes_on_rank, dtype=torch.uint8) * rank
            assert_tensor_equality(
                tensor_a=node_partition_book[node_ids],
                tensor_b=expected_node_pidx,
            )

            edge_ids = graph.edge_ids

            if target_node_type == ITEM_NODE_TYPE:
                # If the target_node_type is ITEM, we expect there to be twice as many edges as nodes since item node is the destination node of two edges.
                self.assertEqual(
                    edge_ids.size(0),
                    node_ids.size(0) * 2,
                )
            else:
                # Otherwise, we expect there to be the same number of edges and nodes, as each user node is the source and destination node of exactly one edge.
                self.assertEqual(
                    edge_ids.size(0),
                    node_ids.size(0),
                )

            # We expect the edge ids on the current rank to be correctly specified in the edge partition book. For example, if we are on rank 0 with edge ids [2, 4, 6], we expect
            # edge_partition_book[[2, 4, 6]] = 0.
            expected_edge_pidx = (
                torch.ones(
                    edge_ids.size(0),
                    dtype=torch.uint8,
                )
                * rank
            )
            assert_tensor_equality(
                tensor_a=edge_partition_book[edge_ids],
                tensor_b=expected_edge_pidx,
            )

    def _assert_node_feature_outputs(
        self,
        rank: int,
        is_heterogeneous: bool,
        edge_assign_strategy: EdgeAssignStrategy,
        output_graph: Union[GraphPartitionData, Dict[EdgeType, GraphPartitionData]],
        output_node_feat: Union[
            FeaturePartitionData, Dict[NodeType, FeaturePartitionData]
        ],
        expected_node_types: List[NodeType],
        expected_edge_types: List[EdgeType],
    ) -> None:
        """
        Checks correctness for node feature outputs of partitioning
        Args:
            rank (int): Rank from current output
            is_heterogeneous (bool): Whether the output is expected to be homogeneous or heterogeneous
            edge_assign_strategy (EdgeAssignStrategy): Whether to partion edges according to the partition book of the source node or destination node
            output_graph: (Union[GraphPartitionData, Dict[EdgeType, GraphPartitionData]]): Output edge indices and ids from partitioning, either a GraphPartitionData if homogeneous or a Dict[EdgeType, GraphPartitionData] if heterogeneous
            output_node_feat (Union[FeaturePartitionData, Dict[NodeType, FeaturePartitionData]]): Output node features from partitioning, either a FeaturePartitionData if homogeneous or a Dict[NodeType, FeaturePartitionData] if heterogeneous
            expected_node_types (List[NodeType]): Expected node types for heterogeneous input
            expected_edge_types (List[EdgeType]): Expected edge types for heterogeneous input
        """
        self._assert_data_type_correctness(
            output_data=output_graph,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )
        self._assert_data_type_correctness(
            output_data=output_node_feat,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_node_types,
        )

        # To unify logic between homogeneous and heterogeneous cases, we define an iterable which we'll loop over.
        # Each iteration contains an EdgeType and a graph consisting of edge indices and ids.
        entity_iterable: Iterable[Tuple[EdgeType, GraphPartitionData]]
        if is_heterogeneous:
            assert isinstance(
                output_graph, abc.Mapping
            ), "Homogeneous output detected from node features for heterogeneous input"
            entity_iterable = list(output_graph.items())
        else:
            assert isinstance(
                output_graph, GraphPartitionData
            ), "Heterogeneous output detected from node features for homogeneous input"
            entity_iterable = [(USER_TO_USER_EDGE_TYPE, output_graph)]

        for edge_type, graph in entity_iterable:
            node_feat: FeaturePartitionData
            node_ids: torch.Tensor
            if edge_assign_strategy == EdgeAssignStrategy.BY_SOURCE_NODE:
                target_node_type = edge_type.src_node_type
                node_ids = torch.unique(graph.edge_index[0])
            else:
                target_node_type = edge_type.dst_node_type
                node_ids = torch.unique(graph.edge_index[1])

            num_nodes_on_rank: int = RANK_TO_NODE_TYPE_TYPE_TO_NUM_NODES[rank][
                target_node_type
            ]

            if is_heterogeneous:
                assert isinstance(
                    output_node_feat, abc.Mapping
                ), "Found homogeneous node features for heterogeneous input"
                node_feat = output_node_feat[target_node_type]
            else:
                assert isinstance(
                    output_node_feat, FeaturePartitionData
                ), "Found heterogeneous node features for homogeneous input"
                node_feat = output_node_feat

            # We expect the number of output node features to be the same as the number of nodes input to the partitioner on the current rank, as the input and output node ids per rank
            # should both be equal in number across all ranks, meaning that node features are also equal in number.
            # This is because each node id is the source node of exactly one edge in the mocked graph.
            self.assertEqual(
                node_feat.feats.size(0),
                num_nodes_on_rank,
            )
            # We expect the node ids on the current rank from the graph to be the same as the node ids on the current rank from the features
            assert_tensor_equality(tensor_a=node_ids, tensor_b=node_feat.ids, dim=0)
            # We expect the shape of the node features to be equal to the expected node feature dimension
            self.assertEqual(
                node_feat.feats.size(1),
                NODE_TYPE_TO_FEATURE_DIMENSION_MAP[target_node_type],
            )
            # We expect the value of each node feature to be equal to its corresponding node id / 10 on the currently mocked input
            for idx, n_id in enumerate(node_feat.ids):
                assert_tensor_equality(
                    tensor_a=node_feat.feats[idx],
                    tensor_b=torch.ones(
                        NODE_TYPE_TO_FEATURE_DIMENSION_MAP[target_node_type],
                        dtype=torch.float32,
                    )
                    * n_id
                    * 0.1,
                )

    def _assert_edge_feature_outputs(
        self,
        rank: int,
        is_heterogeneous: bool,
        edge_assign_strategy: EdgeAssignStrategy,
        output_graph: Union[GraphPartitionData, Dict[EdgeType, GraphPartitionData]],
        output_edge_feat: Union[
            FeaturePartitionData, Dict[EdgeType, FeaturePartitionData]
        ],
        expected_edge_types: List[EdgeType],
    ) -> None:
        """
        Checks correctness for edge feature outputs of partitioning
        Args:
            rank (int): Rank from current output
            is_heterogeneous (bool): Whether the output is expected to be homogeneous or heterogeneous
            edge_assign_strategy (EdgeAssignStrategy): Whether to partion edges according to the partition book of the source node or destination node
            output_graph: (Union[GraphPartitionData, Dict[EdgeType, GraphPartitionData]]): Output edge indices and ids from partitioning, either a GraphPartitionData if homogeneous or a Dict[EdgeType, GraphPartitionData] if heterogeneous
            output_edge_feat (Union[FeaturePartitionData, Dict[EdgeType, FeaturePartitionData]]): Output node features from partitioning, either a FeaturePartitionData if homogeneous or a Dict[EdgeType, FeaturePartitionData] if heterogeneous
            expected_edge_types (List[EdgeType]): Expected edge types for heterogeneous input
        """
        self._assert_data_type_correctness(
            output_data=output_graph,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )
        self._assert_data_type_correctness(
            output_data=output_edge_feat,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )

        # To unify logic between homogeneous and heterogeneous cases, we define an iterable which we'll loop over.
        # Each iteration contains an EdgeType, a feature object containing edge features and edge ids, and a graph consisting of edge indices and ids.
        entity_iterable: Iterable[
            Tuple[EdgeType, FeaturePartitionData, GraphPartitionData]
        ]
        if is_heterogeneous:
            assert isinstance(
                output_edge_feat, abc.Mapping
            ), "Homogeneous output detected from edge features for heterogeneous input"
            assert isinstance(
                output_graph, abc.Mapping
            ), "Homogeneous output detected from graph for heterogeneous input"
            entity_iterable = [
                item
                for item in zip(
                    MOCKED_HETEROGENEOUS_EDGE_TYPES,
                    output_edge_feat.values(),
                    output_graph.values(),
                )
            ]
        else:
            assert isinstance(
                output_edge_feat, FeaturePartitionData
            ), "Heterogeneous output detected from edge features for homogeneous input"
            assert isinstance(
                output_graph, GraphPartitionData
            ), "Heterogeneous output detected from graph for homogeneous input"

            entity_iterable = [(USER_TO_USER_EDGE_TYPE, output_edge_feat, output_graph)]

        for edge_type, edge_feat, graph in entity_iterable:
            if edge_assign_strategy == EdgeAssignStrategy.BY_SOURCE_NODE:
                target_node_type = edge_type.src_node_type
            else:
                target_node_type = edge_type.dst_node_type

            num_nodes_on_rank: int = RANK_TO_NODE_TYPE_TYPE_TO_NUM_NODES[rank][
                target_node_type
            ]

            # We expect the number of edge feats on the current rank to be the same as the number of input nodes to the partitioner on the current rank. This is because
            # the number of edges should be equal to the number of target nodes assigned to each rank and the number of edge feats should be equal to the number of edges.

            if target_node_type == ITEM_NODE_TYPE:
                # If the target_node_type is ITEM, we expect there to be twice as many edges as nodes since item node is the destination node of two edges,
                # and therefore twice as many edge features
                self.assertEqual(edge_feat.feats.size(0), num_nodes_on_rank * 2)
            else:
                # Otherwise, we expect there to be the same number of edge features and nodes, as each user node is the source and destination node of exactly one edge.
                self.assertEqual(edge_feat.feats.size(0), num_nodes_on_rank)

            # We expect the edge ids on the current rank from the graph to be the same as the edge ids on the current rank from the features
            assert_tensor_equality(
                tensor_a=graph.edge_ids, tensor_b=edge_feat.ids, dim=0
            )

            # We expect the shape of the edge features to be equal to the expected edge feature dimension
            self.assertEqual(
                edge_feat.feats.size(1), EDGE_TYPE_TO_FEATURE_DIMENSION_MAP[edge_type]
            )

            # We expect the value of each edge feature to be equal to its corresponding edge id / 10 on the currently mocked input
            for idx, e_id in enumerate(edge_feat.ids):
                assert_tensor_equality(
                    tensor_a=edge_feat.feats[idx],
                    tensor_b=torch.ones(
                        EDGE_TYPE_TO_FEATURE_DIMENSION_MAP[edge_type],
                        dtype=torch.float32,
                    )
                    * e_id
                    * 0.1,
                )

    def _assert_label_outputs(
        self,
        rank: int,
        is_heterogeneous: bool,
        edge_assign_strategy: EdgeAssignStrategy,
        output_node_partition_book: Union[torch.Tensor, Dict[NodeType, torch.Tensor]],
        output_labeled_edge_index: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
        expected_edge_types: List[EdgeType],
    ) -> None:
        """
        Checks correctness for labeled outputs of partitioning
        Args:
            rank (int): Rank from current output
            is_heterogeneous (bool): Whether the output is expected to be homogeneous or heterogeneous
            edge_assign_strategy (EdgeAssignStrategy): Whether to partion edges according to the partition book of the source node or destination node
            output_node_partition_book: (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): Node Partition Book from partitioning, either a Tensor if homogeneous or a Dict[NodeType, Tensor] if heterogeneous
            output_labeled_edge_index (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Output labeled edges from partitioning, either a FeaturePartitionData if homogeneous or a Dict[EdgeType, FeaturePartitionData] if heterogeneous
            expected_edge_types (List[EdgeType]): Expected edge types for heterogeneous input
        """

        self._assert_data_type_correctness(
            output_data=output_labeled_edge_index,
            is_heterogeneous=is_heterogeneous,
            expected_entity_types=expected_edge_types,
        )

        entity_iterable: Iterable[Tuple[EdgeType, torch.Tensor]]
        if is_heterogeneous:
            assert isinstance(
                output_labeled_edge_index, abc.Mapping
            ), "Homogeneous output detected from labels for heterogeneous input"
            entity_iterable = list(output_labeled_edge_index.items())
        else:
            assert isinstance(
                output_labeled_edge_index, torch.Tensor
            ), "Heterogeneous output detected from labels for homogeneous input"
            entity_iterable = [(USER_TO_USER_EDGE_TYPE, output_labeled_edge_index)]

        for edge_type, labeled_edge_index in entity_iterable:
            node_partition_book: torch.Tensor

            if edge_assign_strategy == EdgeAssignStrategy.BY_SOURCE_NODE:
                target_node_type = edge_type.src_node_type
                target_nodes = labeled_edge_index[0]
            else:
                target_node_type = edge_type.dst_node_type
                target_nodes = labeled_edge_index[1]

            if is_heterogeneous:
                assert isinstance(
                    output_node_partition_book, abc.Mapping
                ), "Homogeneous node partition book detected for heterogeneous input"
                node_partition_book = output_node_partition_book[target_node_type]
            else:
                assert isinstance(
                    output_node_partition_book, torch.Tensor
                ), "Heterogeneous node partition book detected for homogeneous input"
                node_partition_book = output_node_partition_book

            # Since we don't have as many labeled edges as nodes, we cannot guarantee the number of labeled edges per rank.
            # As a result, we validate correctness by checking that the labeled edge indices on the current rank align with the
            # node partition book
            expect_edge_pidx = (
                torch.ones(
                    target_nodes.size(0),
                    dtype=torch.uint8,
                )
                * rank
            )

            assert_tensor_equality(node_partition_book[target_nodes], expect_edge_pidx)

    @parameterized.expand(
        [
            param(
                "Homogeneous Partitioning by Source Node - Register All Entites together through Constructor",
                is_heterogeneous=False,
                input_data_strategy=InputDataStrategy.REGISTER_ALL_ENTITIES_TOGETHER,
                edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
            ),
            param(
                "Heterogeneous Partitioning By Source Node - Register All Entites together through Constructor",
                is_heterogeneous=True,
                input_data_strategy=InputDataStrategy.REGISTER_ALL_ENTITIES_TOGETHER,
                edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
            ),
            param(
                "Homogeneous Partitioning By Dest Node- Register All Entites together through Constructor",
                is_heterogeneous=False,
                input_data_strategy=InputDataStrategy.REGISTER_ALL_ENTITIES_TOGETHER,
                edge_assign_strategy=EdgeAssignStrategy.BY_DESTINATION_NODE,
            ),
            param(
                "Heterogeneous Partitioning By Dest Node- Register All Entites together through Constructor",
                is_heterogeneous=True,
                input_data_strategy=InputDataStrategy.REGISTER_ALL_ENTITIES_TOGETHER,
                edge_assign_strategy=EdgeAssignStrategy.BY_DESTINATION_NODE,
            ),
            param(
                "Homogeneous Partitioning By Source Node - Register All Entites separately through register functions",
                is_heterogeneous=False,
                input_data_strategy=InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
                edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
            ),
            param(
                "Homogeneous Partitioning By Source Node - Register minimal entities separately through register functions",
                is_heterogeneous=False,
                input_data_strategy=InputDataStrategy.REGISTER_MINIMAL_ENTITIES_SEPARATELY,
                edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
            ),
        ]
    )
    def test_partitioning_correctness(
        self,
        _,
        is_heterogeneous: bool,
        input_data_strategy: InputDataStrategy,
        edge_assign_strategy: EdgeAssignStrategy,
    ) -> None:
        """
        Tests partitioning functionality and correctness on mocked inputs
        Args:
            is_heterogeneous (bool): Whether homogeneous or heterogeneous inputs should be used
            input_data_strategy (InputDataStrategy): Strategy for registering inputs to the partitioner
            edge_assign_strategy (EdgeAssignStrategy): Whether to partion edges according to the partition book of the source node or destination node
        """
        master_port = glt.utils.get_free_port(self._master_ip_address)

        manager = Manager()
        output_dict: MutableMapping[int, PartitionOutput] = manager.dict()

        mocked_input_graph = RANK_TO_MOCKED_GRAPH

        mp.spawn(
            run_distributed_partitioner,
            args=(
                output_dict,
                is_heterogeneous,
                mocked_input_graph,
                edge_assign_strategy,
                self._master_ip_address,
                master_port,
                input_data_strategy,
            ),
            nprocs=MOCKED_NUM_PARTITIONS,
            join=True,
        )

        unified_output_edge_index: Dict[EdgeType, List[torch.Tensor]] = defaultdict(
            list
        )

        unified_output_node_feat: Dict[NodeType, List[torch.Tensor]] = defaultdict(list)

        unified_output_edge_feat: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)

        unified_output_pos_label: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)

        unified_output_neg_label: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)

        for rank, partition_output in output_dict.items():
            self._assert_graph_outputs(
                rank=rank,
                is_heterogeneous=is_heterogeneous,
                edge_assign_strategy=edge_assign_strategy,
                output_node_partition_book=partition_output.node_partition_book,
                output_edge_partition_book=partition_output.edge_partition_book,
                output_edge_index=partition_output.partitioned_edge_index,
                expected_node_types=MOCKED_HETEROGENEOUS_NODE_TYPES,
                expected_edge_types=MOCKED_HETEROGENEOUS_EDGE_TYPES,
            )
            if isinstance(partition_output.partitioned_edge_index, abc.Mapping):
                for edge_type, graph in partition_output.partitioned_edge_index.items():
                    unified_output_edge_index[edge_type].append(graph.edge_index)
            else:
                graph = partition_output.partitioned_edge_index
                unified_output_edge_index[USER_TO_USER_EDGE_TYPE].append(
                    graph.edge_index
                )

            if (
                input_data_strategy
                == InputDataStrategy.REGISTER_MINIMAL_ENTITIES_SEPARATELY
            ):
                self.assertIsNone(partition_output.partitioned_edge_features)
                self.assertIsNone(partition_output.partitioned_node_features)
                self.assertIsNone(partition_output.partitioned_positive_labels)
                self.assertIsNone(partition_output.partitioned_negative_labels)
            else:
                assert (
                    partition_output.partitioned_node_features is not None
                ), f"Must partition node features for strategy {input_data_strategy.value}"
                assert (
                    partition_output.partitioned_edge_features is not None
                ), f"Must partition edge features for strategy {input_data_strategy.value}"

                assert (
                    partition_output.partitioned_positive_labels is not None
                ), f"Must partition positive labels for strategy {input_data_strategy.value}"

                assert (
                    partition_output.partitioned_negative_labels is not None
                ), f"Must partition negative labels for strategy {input_data_strategy.value}"

                self._assert_node_feature_outputs(
                    rank=rank,
                    is_heterogeneous=is_heterogeneous,
                    edge_assign_strategy=edge_assign_strategy,
                    output_graph=partition_output.partitioned_edge_index,
                    output_node_feat=partition_output.partitioned_node_features,
                    expected_node_types=MOCKED_HETEROGENEOUS_NODE_TYPES,
                    expected_edge_types=MOCKED_HETEROGENEOUS_EDGE_TYPES,
                )

                if isinstance(partition_output.partitioned_node_features, abc.Mapping):
                    for (
                        node_type,
                        node_features,
                    ) in partition_output.partitioned_node_features.items():
                        unified_output_node_feat[node_type].append(node_features.feats)
                else:
                    unified_output_node_feat[USER_NODE_TYPE].append(
                        partition_output.partitioned_node_features.feats
                    )

                self._assert_edge_feature_outputs(
                    rank=rank,
                    is_heterogeneous=is_heterogeneous,
                    edge_assign_strategy=edge_assign_strategy,
                    output_graph=partition_output.partitioned_edge_index,
                    output_edge_feat=partition_output.partitioned_edge_features,
                    expected_edge_types=MOCKED_HETEROGENEOUS_EDGE_TYPES,
                )

                if isinstance(partition_output.partitioned_edge_features, abc.Mapping):
                    for (
                        edge_type,
                        edge_features,
                    ) in partition_output.partitioned_edge_features.items():
                        unified_output_edge_feat[edge_type].append(edge_features.feats)
                else:
                    unified_output_edge_feat[USER_TO_USER_EDGE_TYPE].append(
                        partition_output.partitioned_edge_features.feats
                    )

                # Currently, we always partition labeled edge indices by their source node
                self._assert_label_outputs(
                    rank=rank,
                    is_heterogeneous=is_heterogeneous,
                    output_node_partition_book=partition_output.node_partition_book,
                    edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
                    output_labeled_edge_index=partition_output.partitioned_positive_labels,
                    expected_edge_types=MOCKED_HETEROGENEOUS_EDGE_TYPES,
                )

                if isinstance(
                    partition_output.partitioned_positive_labels, abc.Mapping
                ):
                    for (
                        edge_type,
                        pos_edge_label,
                    ) in partition_output.partitioned_positive_labels.items():
                        unified_output_pos_label[edge_type].append(pos_edge_label)
                else:
                    unified_output_pos_label[USER_TO_USER_EDGE_TYPE].append(
                        partition_output.partitioned_positive_labels
                    )

                self._assert_label_outputs(
                    rank=rank,
                    is_heterogeneous=is_heterogeneous,
                    output_node_partition_book=partition_output.node_partition_book,
                    edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
                    output_labeled_edge_index=partition_output.partitioned_negative_labels,
                    expected_edge_types=MOCKED_HETEROGENEOUS_EDGE_TYPES,
                )

                if isinstance(
                    partition_output.partitioned_negative_labels, abc.Mapping
                ):
                    for (
                        edge_type,
                        neg_edge_labels,
                    ) in partition_output.partitioned_negative_labels.items():
                        unified_output_neg_label[edge_type].append(neg_edge_labels)
                else:
                    unified_output_neg_label[USER_TO_USER_EDGE_TYPE].append(
                        partition_output.partitioned_negative_labels
                    )

        ## Checking that the union of edge indices across all ranks equals to the full set from the input

        for edge_type in unified_output_edge_index:
            # First, we get the expected edge index from the mocked input for this edge type
            expected_edge_index = MOCKED_UNIFIED_GRAPH.edge_index[edge_type]

            # We combine the output edge indices across all the ranks
            output_edge_index = torch.cat(unified_output_edge_index[edge_type], dim=1)

            # Finally, we check that the expected tensor and output tensor have the same columns, which is achieved by setting the shuffle dimension to 1
            assert_tensor_equality(
                tensor_a=expected_edge_index, tensor_b=output_edge_index, dim=1
            )

        ## Checking for the union of node features across all ranks equals to the full set from the input

        for node_type in unified_output_node_feat:
            # First, we get the expected node features from the mocked input for this node type
            expected_node_feat = MOCKED_UNIFIED_GRAPH.node_features[node_type]

            # We combine the output node features across all the ranks
            output_node_feat = torch.cat(unified_output_node_feat[node_type], dim=0)

            # Finally, we check that the expected tensor and output tensor have the same rows, which is achieved by setting the shuffle dimension to 0
            assert_tensor_equality(
                tensor_a=expected_node_feat, tensor_b=output_node_feat, dim=0
            )

        ## Checking for the union of edge features across all ranks equals to the full set from the input

        for edge_type in unified_output_edge_feat:
            # First, we get the expected edge features from the mocked input for this edge type
            expected_edge_feat = MOCKED_UNIFIED_GRAPH.edge_features[edge_type]

            # We combine the output edge features across all the ranks
            output_edge_feat = torch.cat(unified_output_edge_feat[edge_type], dim=0)

            # Finally, we check that the expected tensor and output tensor have the same rows, which is achieved by setting the shuffle dimension to 0
            assert_tensor_equality(
                tensor_a=expected_edge_feat, tensor_b=output_edge_feat, dim=0
            )

        for edge_type in unified_output_pos_label:
            # First, we get the expected pos edge label from the mocked input for this edge type
            expected_positive_labels = MOCKED_UNIFIED_GRAPH.positive_labels[edge_type]

            # We combine the output pos labels across all the ranks
            output_pos_label = torch.cat(unified_output_pos_label[edge_type], dim=1)

            # Finally, we check that the expected tensor and output tensor have the same rows, which is achieved by setting the shuffle dimension to 1
            assert_tensor_equality(
                tensor_a=expected_positive_labels, tensor_b=output_pos_label, dim=1
            )

        for edge_type in unified_output_neg_label:
            # First, we get the expected neg edge label from the mocked input for this edge type
            expected_negative_labels = MOCKED_UNIFIED_GRAPH.negative_labels[edge_type]

            # We combine the output neg labels across all the ranks
            output_neg_label = torch.cat(unified_output_neg_label[edge_type], dim=1)

            # Finally, we check that the expected tensor and output tensor have the same rows, which is achieved by setting the shuffle dimension to 1
            assert_tensor_equality(
                tensor_a=expected_negative_labels, tensor_b=output_neg_label, dim=1
            )

    def test_partitioning_failure(self) -> None:
        master_port = glt.utils.get_free_port(self._master_ip_address)
        rank = 0

        input_graph = RANK_TO_MOCKED_GRAPH[rank]

        node_ids = input_graph.node_ids[USER_NODE_TYPE]
        node_features = input_graph.node_features[USER_NODE_TYPE]

        init_worker_group(world_size=1, rank=rank)
        init_rpc(
            master_addr=self._master_ip_address,
            master_port=master_port,
            num_rpc_threads=4,
        )

        partitioner = DistLinkPredictionDataPartitioner(
            edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
        )

        # Assert that calling partition without any registering raises error
        with self.subTest(partitioner=partitioner):
            with self.assertRaisesRegex(
                AssertionError, "Must have registered nodes prior to partitioning them"
            ):
                partitioner.partition()

        partitioner.register_node_ids(node_ids=node_ids)
        partitioner.register_node_features(node_features=node_features)

        # Assert that calling partition without registering edges raises error. This fails because we require edges be registered prior to calling .partition()
        with self.subTest(partitioner=partitioner):
            with self.assertRaisesRegex(
                AssertionError, "Must have registered edges prior to partitioning them"
            ):
                partitioner.partition()

        node_partition_book = partitioner.partition_node()
        partitioner.partition_node_features(node_partition_book)

        # Assert that calling partition_node after calling partition() and partition_node_features raises error, as the inputs have been cleaned up at this point
        with self.subTest(partitioner=partitioner):
            with self.assertRaisesRegex(
                AssertionError, "Must have registered nodes prior to partitioning them"
            ):
                partitioner.partition_node()

        partitioner = DistLinkPredictionDataPartitioner(
            edge_assign_strategy=EdgeAssignStrategy.BY_SOURCE_NODE,
        )
        empty_node_ids = torch.empty(0)
        empty_edge_index = torch.empty((2, 0))
        empty_node_features = torch.empty((0, 5))
        empty_edge_features = torch.empty((0, 10))
        empty_pos_label = torch.empty((2, 0))
        empty_neg_label = torch.empty((2, 0))

        # Test partitioning with empty node_ids, empty node_feats, empty edge_feats, and empty edge index
        partitioner.register_node_ids(node_ids=empty_node_ids)
        partitioner.register_edge_index(edge_index=empty_edge_index)
        partitioner.register_node_features(node_features=empty_node_features)
        partitioner.register_edge_features(edge_features=empty_edge_features)
        partitioner.register_labels(label_edge_index=empty_pos_label, is_positive=True)
        partitioner.register_labels(label_edge_index=empty_neg_label, is_positive=False)

        partitioned_output = partitioner.partition()
        assert not isinstance(
            partitioned_output.partitioned_node_features, abc.Mapping
        ) and not isinstance(
            partitioned_output.partitioned_edge_features, abc.Mapping
        ), "Got heterogeneous features, but expected homogeneous output"

        assert (
            partitioned_output.partitioned_node_features is not None
            and partitioned_output.partitioned_edge_features is not None
        ), "Features should not be None"

        assert (
            partitioned_output.partitioned_node_features.feats.shape
            == empty_node_features.shape
        ), f"Node Features should be empty, but got shape {partitioned_output.partitioned_node_features.feats.shape}"

        assert (
            partitioned_output.partitioned_edge_features.feats.shape
            == empty_edge_features.shape
        ), f"Edge Features should be empty, but got shape {partitioned_output.partitioned_edge_features.feats.shape}"

        # Assert that calling partition() twice in a row on registered input raises error
        with self.subTest(partitioner=partitioner):
            with self.assertRaisesRegex(
                AssertionError, "Must have registered nodes prior to partitioning them"
            ):
                partitioner.partition()


if __name__ == "__main__":
    unittest.main()
