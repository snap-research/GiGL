import unittest
from collections import abc
from typing import Any, MutableMapping, Optional, Union

import graphlearn_torch as glt
import torch
from parameterized import param, parameterized
from torch.multiprocessing import Manager
from torch.testing import assert_close

from gigl.distributed import (
    DistLinkPredictionDataPartitioner,
    DistLinkPredictionDataset,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.distributed import PartitionOutput
from gigl.utils.data_splitters import NodeAnchorLinkSplitter
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)

_PARTITIONER_ERROR_MESSAGE = "Test partitioner subclass not implemented"
_DATASET_ERROR_MESSAGE = "Test dataset subclass not implemented"


class _PartitionerTestingSubclass(DistLinkPredictionDataPartitioner):
    def partition(self) -> PartitionOutput:
        # This subclass raises a NotImplementedError so that we can identify this custom logic being used in place of
        # the base DistLinkPredicitonDataPartitioner.partition() logic.
        raise NotImplementedError(_PARTITIONER_ERROR_MESSAGE)


class _DatasetTestingSubclass(DistLinkPredictionDataset):
    def build(
        self,
        partition_output: PartitionOutput,
        splitter: Optional[NodeAnchorLinkSplitter] = None,
    ) -> None:
        # This subclass raises a NotImplementedError so that we can identify this custom logic being used in place of
        # the base DistLinkPredictionDataset.build() logic.
        raise NotImplementedError(_DATASET_ERROR_MESSAGE)


class _FakeSplitter:
    def __init__(
        self,
        splits: Union[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            dict[EdgeType, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ],
    ):
        self.splits = splits

    def __call__(self, edge_index):
        return self.splits


_USER = NodeType("user")
_STORY = NodeType("story")


class LoadAndBuildDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"
        self._world_size = 1
        self._num_rpc_threads = 4

    def assert_tensor_equal(
        self,
        actual: Optional[Union[torch.Tensor, abc.Mapping[Any, torch.Tensor]]],
        expected: Optional[Union[torch.Tensor, abc.Mapping[Any, torch.Tensor]]],
    ):
        if type(actual) != type(expected):
            self.fail(f"Expected type {type(expected)} but got {type(actual)}")
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual.keys():
                assert_close(actual[key], expected[key], atol=0, rtol=0)
        elif isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            assert_close(actual, expected, atol=0, rtol=0)

    def test_load_and_build_dataset(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
        )

        self.assertIsNone(dataset.train_node_ids)
        self.assertIsNone(dataset.val_node_ids)
        self.assertIsNone(dataset.test_node_ids)
        self.assertIsInstance(dataset.node_ids, torch.Tensor)

    def test_load_build_and_split_dataset(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
            splitter=_FakeSplitter(
                (
                    torch.tensor([1000]),
                    torch.tensor([2000, 3000]),
                    torch.tensor([3000, 4000, 5000]),
                ),
            ),
        )

        self.assert_tensor_equal(dataset.train_node_ids, torch.tensor([1000]))
        self.assert_tensor_equal(dataset.val_node_ids, torch.tensor([2000, 3000]))
        self.assert_tensor_equal(
            dataset.test_node_ids, torch.tensor([3000, 4000, 5000])
        )
        # Check that the node ids have *all* node ids, including nodes not included in train, val, and test.
        self.assert_tensor_equal(
            dataset.node_ids,
            torch.tensor(
                [
                    1000,
                    2000,
                    3000,
                    3000,
                    4000,
                    5000,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                ]
            ),
        )

    @parameterized.expand(
        [
            param(
                "One supervision edge type",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000]),
                        torch.tensor([3000]),
                    )
                },
                expected_train_node_ids={_USER: torch.tensor([1000])},
                expected_val_node_ids={_USER: torch.tensor([2000])},
                expected_test_node_ids={_USER: torch.tensor([3000])},
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
            param(
                "One supervision edge type - different numbers of train-test-val",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000, 3000]),
                        torch.tensor([3000, 4000, 5000]),
                    )
                },
                expected_train_node_ids={_USER: torch.tensor([1000])},
                expected_val_node_ids={_USER: torch.tensor([2000, 3000])},
                expected_test_node_ids={_USER: torch.tensor([3000, 4000, 5000])},
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            3000,
                            4000,
                            5000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
            param(
                "Two supervision edge types - two target node types",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000]),
                        torch.tensor([3000]),
                    ),
                    _STORY: (
                        torch.tensor([4000]),
                        torch.tensor([5000]),
                        torch.tensor([6000]),
                    ),
                },
                expected_train_node_ids={
                    _USER: torch.tensor([1000]),
                    _STORY: torch.tensor([4000]),
                },
                expected_val_node_ids={
                    _USER: torch.tensor([2000]),
                    _STORY: torch.tensor([5000]),
                },
                expected_test_node_ids={
                    _USER: torch.tensor([3000]),
                    _STORY: torch.tensor([6000]),
                },
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            4000,
                            5000,
                            6000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
        ]
    )
    def test_load_build_and_split_dataset_heterogeneous(
        self,
        _,
        splits,
        expected_train_node_ids,
        expected_val_node_ids,
        expected_test_node_ids,
        expected_node_ids,
    ):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
            splitter=_FakeSplitter(splits),
        )

        self.assert_tensor_equal(dataset.train_node_ids, expected_train_node_ids)
        self.assert_tensor_equal(dataset.val_node_ids, expected_val_node_ids)
        self.assert_tensor_equal(dataset.test_node_ids, expected_test_node_ids)
        # Check that the node ids have *all* node ids, including nodes not included in train, val, and test.
        self.assert_tensor_equal(dataset.node_ids, expected_node_ids)

    @parameterized.expand(
        [
            param(
                "Tests custom Partitioner subclasses can be used in `load_and_build_partitioned_dataset`",
                should_use_custom_partitioner=True,
                should_use_custom_dataset=False,
                expected_error_message=_PARTITIONER_ERROR_MESSAGE,
            ),
            param(
                "Tests custom Dataset subclasses can be used in `load_and_build_partitioned_dataset`",
                should_use_custom_partitioner=False,
                should_use_custom_dataset=True,
                expected_error_message=_DATASET_ERROR_MESSAGE,
            ),
        ]
    )
    def test_load_and_build_subclasses(
        self,
        _,
        should_use_custom_partitioner: bool,
        should_use_custom_dataset: bool,
        expected_error_message: str,
    ) -> None:
        master_port = glt.utils.get_free_port(self._master_ip_address)

        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        if should_use_custom_partitioner:
            partitioner = _PartitionerTestingSubclass()
        else:
            partitioner = None

        if should_use_custom_dataset:
            dataset = _DatasetTestingSubclass(rank=0, world_size=1, edge_dir="in")
        else:
            dataset = None

        with self.assertRaisesRegex(NotImplementedError, expected_error_message):
            run_distributed_dataset(
                rank=0,
                world_size=self._world_size,
                mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                output_dict=output_dict,
                should_load_tensors_in_parallel=True,
                master_ip_address=self._master_ip_address,
                master_port=master_port,
                partitioner=partitioner,
                dataset=dataset,
            )


if __name__ == "__main__":
    unittest.main()
