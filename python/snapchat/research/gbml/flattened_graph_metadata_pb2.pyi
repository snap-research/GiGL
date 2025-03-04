"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class SupervisedNodeClassificationOutput(google.protobuf.message.Message):
    """Stores SupervisedNodeClassificationSample-relevant output"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    UNLABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    labeled_tfrecord_uri_prefix: builtins.str
    """GCS prefix which can be used to glob the TFRecord dataset."""
    unlabeled_tfrecord_uri_prefix: builtins.str
    def __init__(
        self,
        *,
        labeled_tfrecord_uri_prefix: builtins.str = ...,
        unlabeled_tfrecord_uri_prefix: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["labeled_tfrecord_uri_prefix", b"labeled_tfrecord_uri_prefix", "unlabeled_tfrecord_uri_prefix", b"unlabeled_tfrecord_uri_prefix"]) -> None: ...

global___SupervisedNodeClassificationOutput = SupervisedNodeClassificationOutput

class NodeAnchorBasedLinkPredictionOutput(google.protobuf.message.Message):
    """Stores NodeAnchorBasedLinkPredictionSample-relevant output"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class NodeTypeToRandomNegativeTfrecordUriPrefixEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    NODE_TYPE_TO_RANDOM_NEGATIVE_TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    tfrecord_uri_prefix: builtins.str
    """GCS prefix which can be used to glob the TFRecord dataset."""
    @property
    def node_type_to_random_negative_tfrecord_uri_prefix(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]:
        """Rooted subgraphs for each type of nodes; besides training, also used for inference as these are just subgraphs for each node"""
    def __init__(
        self,
        *,
        tfrecord_uri_prefix: builtins.str = ...,
        node_type_to_random_negative_tfrecord_uri_prefix: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["node_type_to_random_negative_tfrecord_uri_prefix", b"node_type_to_random_negative_tfrecord_uri_prefix", "tfrecord_uri_prefix", b"tfrecord_uri_prefix"]) -> None: ...

global___NodeAnchorBasedLinkPredictionOutput = NodeAnchorBasedLinkPredictionOutput

class SupervisedLinkBasedTaskOutput(google.protobuf.message.Message):
    """Stores SupervisedLinkBasedTaskSample-relevant output"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    UNLABELED_TFRECORD_URI_PREFIX_FIELD_NUMBER: builtins.int
    labeled_tfrecord_uri_prefix: builtins.str
    """GCS prefix which can be used to glob the TFRecord dataset."""
    unlabeled_tfrecord_uri_prefix: builtins.str
    def __init__(
        self,
        *,
        labeled_tfrecord_uri_prefix: builtins.str = ...,
        unlabeled_tfrecord_uri_prefix: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["labeled_tfrecord_uri_prefix", b"labeled_tfrecord_uri_prefix", "unlabeled_tfrecord_uri_prefix", b"unlabeled_tfrecord_uri_prefix"]) -> None: ...

global___SupervisedLinkBasedTaskOutput = SupervisedLinkBasedTaskOutput

class FlattenedGraphMetadata(google.protobuf.message.Message):
    """Stores flattened graph metadata output by SubgraphSampler"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUPERVISED_NODE_CLASSIFICATION_OUTPUT_FIELD_NUMBER: builtins.int
    NODE_ANCHOR_BASED_LINK_PREDICTION_OUTPUT_FIELD_NUMBER: builtins.int
    SUPERVISED_LINK_BASED_TASK_OUTPUT_FIELD_NUMBER: builtins.int
    @property
    def supervised_node_classification_output(self) -> global___SupervisedNodeClassificationOutput:
        """indicates the output is of SupervisedNodeClassificationSamples"""
    @property
    def node_anchor_based_link_prediction_output(self) -> global___NodeAnchorBasedLinkPredictionOutput:
        """indicates the output is of NodeAnchorBasedLinkPredictionSamples"""
    @property
    def supervised_link_based_task_output(self) -> global___SupervisedLinkBasedTaskOutput:
        """indicates the output is of SupervisedLinkBasedTaskSamples"""
    def __init__(
        self,
        *,
        supervised_node_classification_output: global___SupervisedNodeClassificationOutput | None = ...,
        node_anchor_based_link_prediction_output: global___NodeAnchorBasedLinkPredictionOutput | None = ...,
        supervised_link_based_task_output: global___SupervisedLinkBasedTaskOutput | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["node_anchor_based_link_prediction_output", b"node_anchor_based_link_prediction_output", "output_metadata", b"output_metadata", "supervised_link_based_task_output", b"supervised_link_based_task_output", "supervised_node_classification_output", b"supervised_node_classification_output"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["node_anchor_based_link_prediction_output", b"node_anchor_based_link_prediction_output", "output_metadata", b"output_metadata", "supervised_link_based_task_output", b"supervised_link_based_task_output", "supervised_node_classification_output", b"supervised_node_classification_output"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["output_metadata", b"output_metadata"]) -> typing_extensions.Literal["supervised_node_classification_output", "node_anchor_based_link_prediction_output", "supervised_link_based_task_output"] | None: ...

global___FlattenedGraphMetadata = FlattenedGraphMetadata
