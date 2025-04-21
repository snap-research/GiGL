import unittest

from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.distributed import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    to_heterogeneous_edge,
    to_heterogeneous_node,
    to_homogeneous,
)


class DistributedTypesTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "custom_node_type",
                {"custom_node_type": "value"},
                {"custom_node_type": "value"},
            ),
            param(
                "default_node_type", "value", {DEFAULT_HOMOGENEOUS_NODE_TYPE: "value"}
            ),
        ]
    )
    def test_to_hetergeneous_node(self, _, input_value, expected_output):
        self.assertEqual(to_heterogeneous_node(input_value), expected_output)

    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "custom_edge_type",
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
            ),
            param(
                "default_edge_type", "value", {DEFAULT_HOMOGENEOUS_EDGE_TYPE: "value"}
            ),
        ]
    )
    def test_to_hetergeneous_edge(self, _, input_value, expected_output):
        self.assertEqual(to_heterogeneous_edge(input_value), expected_output)

    @parameterized.expand(
        [
            param("none_input", None, None),
            param(
                "single_value_input",
                {EdgeType(NodeType("src"), Relation("rel"), NodeType("dst")): "value"},
                "value",
            ),
            param("direct_value_input", "value", "value"),
        ]
    )
    def test_from_heterogeneous(self, _, input_value, expected_output):
        self.assertEqual(to_homogeneous(input_value), expected_output)

    @parameterized.expand(
        [
            param(
                "multiple_keys_input",
                {NodeType("src"): "src_value", NodeType("dst"): "dst_value"},
            ),
            param(
                "empty_dict_input",
                {},
            ),
        ]
    )
    def test_from_heterogeneous_invalid(self, _, input_value):
        with self.assertRaises(ValueError):
            to_homogeneous(input_value)


if __name__ == "__main__":
    unittest.main()
