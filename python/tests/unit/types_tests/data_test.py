import unittest

import torch
from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.data import LoadedGraphTensors
from gigl.types.distributed import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    NEGATIVE_LABEL_RELATION,
    POSITIVE_LABEL_RELATION,
)


class TestLoadedGraphTensors(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "valid_inputs",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_label=torch.tensor([[0, 2]]),
                negative_label=torch.tensor([[1, 0]]),
                expected_edge_index={
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE: torch.tensor([[0, 1], [1, 2]]),
                    EdgeType(
                        DEFAULT_HOMOGENEOUS_NODE_TYPE,
                        POSITIVE_LABEL_RELATION,
                        DEFAULT_HOMOGENEOUS_NODE_TYPE,
                    ): torch.tensor([[0, 2]]),
                    EdgeType(
                        DEFAULT_HOMOGENEOUS_NODE_TYPE,
                        NEGATIVE_LABEL_RELATION,
                        DEFAULT_HOMOGENEOUS_NODE_TYPE,
                    ): torch.tensor([[1, 0]]),
                },
            ),
        ]
    )
    def test_treat_labels_as_edges_success(
        self,
        name,
        node_ids,
        node_features,
        edge_index,
        edge_features,
        positive_label,
        negative_label,
        expected_edge_index,
    ):
        graph_tensors = LoadedGraphTensors(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        graph_tensors.treat_labels_as_edges()
        self.assertIsNone(graph_tensors.positive_label)
        self.assertIsNone(graph_tensors.negative_label)
        assert isinstance(graph_tensors.edge_index, dict)
        self.assertEqual(graph_tensors.edge_index.keys(), expected_edge_index.keys())
        for edge_type, expected_tensor in expected_edge_index.items():
            torch.testing.assert_close(
                graph_tensors.edge_index[edge_type], expected_tensor
            )

    @parameterized.expand(
        [
            param(
                "missing_labels",
                node_ids=torch.tensor([0, 1, 2]),
                node_features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_features=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                positive_label=None,
                negative_label=None,
                raises=ValueError,
            ),
            param(
                "heterogeneous_inputs",
                node_ids={NodeType("type1"): torch.tensor([0, 1])},
                node_features=None,
                edge_index={
                    EdgeType(
                        NodeType("node1"), Relation("relation"), NodeType("node2")
                    ): torch.tensor([[0, 1]])
                },
                edge_features=None,
                positive_label=torch.tensor([[0, 2]]),
                negative_label=torch.tensor([[1, 0]]),
                raises=ValueError,
            ),
        ]
    )
    def test_treat_labels_as_edges_errors(
        self,
        name,
        node_ids,
        node_features,
        edge_index,
        edge_features,
        positive_label,
        negative_label,
        raises,
    ):
        graph_tensors = LoadedGraphTensors(
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        with self.assertRaises(raises):
            graph_tensors.treat_labels_as_edges()


if __name__ == "__main__":
    unittest.main()
