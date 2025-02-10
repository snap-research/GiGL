from typing import Any, Callable, Iterable, Optional, Tuple, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import tfx_bsl
from apache_beam.pvalue import PBegin, PCollection, PDone
from tensorflow_metadata.proto.v0 import schema_pb2, statistics_pb2
from tensorflow_transform import beam as tft_beam
from tfx_bsl.tfxio.record_based_tfxio import RecordBasedTFXIO

from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.beam.better_tfrecordio import BetterWriteToTFRecord  # type: ignore
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.dataflow import init_beam_pipeline_options
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.data_preprocessor.lib.ingest.reference import (
    DataReference,
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.transform.tf_value_encoder import TFValueEncoder
from gigl.src.data_preprocessor.lib.transform.transformed_features_info import (
    TransformedFeaturesInfo,
)
from gigl.src.data_preprocessor.lib.types import (
    EdgeDataPreprocessingSpec,
    FeatureSpecDict,
    InstanceDict,
    NodeDataPreprocessingSpec,
    TFTensorDict,
)

logger = Logger()


class InstanceDictToTFExample(beam.DoFn):
    """
    Uses a feature spec to process a raw instance dict (read from some tabular data) as a TFExample.  These
    instance dict inputs could allow us to read tabular input data from BQ, GSC or anything else. As long as we
    have a way of yielding instance dicts and parsing them with a feature spec, we should be able to
    transform this data into TFRecords during ingestion, which allows for more efficient operations in TFT.
    See https://www.tensorflow.org/tfx/transform/get_started#the_tfxio_format.
    """

    def __init__(self, feature_spec: FeatureSpecDict):
        self.feature_spec = feature_spec

    def process(self, element: InstanceDict) -> Iterable[bytes]:
        # Each row is a single instance dict from the original tabular input (BQ, GCS, etc.)
        example = dict()
        for key in self.feature_spec.keys():
            # prepare each value associated with a key that appears in the feature_spec.
            # only the instance dict keys the user specifies wanting in the feature_spec will pass through here
            value = element[key]
            if value is None:
                logger.debug(f"Found key {key} with missing value in sample {element}")
            example[key] = TFValueEncoder.encode_value_as_feature(
                value=value, dtype=self.feature_spec[key].dtype
            )
        example_proto = tf.train.Example(features=tf.train.Features(feature=example))
        serialized_proto = example_proto.SerializeToString()
        return [serialized_proto]


class IngestRawFeatures(beam.PTransform):
    # TODO: investigate whether convert to TFXIO is adding overhead instead of speeding things up.
    def __init__(
        self,
        data_reference: DataReference,
        feature_spec: FeatureSpecDict,
        beam_record_tfxio: RecordBasedTFXIO,
    ):
        self.data_reference = data_reference
        self.feature_spec = feature_spec
        self.beam_record_tfxio = beam_record_tfxio

    def expand(self, pbegin: PBegin) -> PCollection[pa.RecordBatch]:
        if not isinstance(pbegin, PBegin):
            raise TypeError(
                f"Input to {IngestRawFeatures.__name__} transform "
                f"must be a PBegin but found {pbegin})"
            )
        return (
            pbegin
            | "Parse raw tabular features into instance dicts."
            >> self.data_reference.yield_instance_dict_ptransform()
            | "Serialize instance dicts to transformed TFExamples"
            >> beam.ParDo(InstanceDictToTFExample(feature_spec=self.feature_spec))
            | "Transformed TFExamples to RecordBatches with TFXIO"
            >> self.beam_record_tfxio.BeamSource()
        )


class GenerateAndVisualizeStats(beam.PTransform):
    def __init__(self, facets_report_uri: GcsUri, stats_output_uri: GcsUri):
        self.facets_report_uri = facets_report_uri
        self.stats_output_uri = stats_output_uri

    def expand(
        self, features: PCollection[pa.RecordBatch]
    ) -> PCollection[statistics_pb2.DatasetFeatureStatisticsList]:
        stats = features | "Generate TFDV statistics" >> tfdv.GenerateStatistics()

        _ = (
            stats
            | "Generate stats visualization"
            >> beam.Map(tfdv.utils.display_util.get_statistics_html)
            | "Write stats Facets report HTML"
            >> beam.io.WriteToText(
                self.facets_report_uri.uri, num_shards=1, shard_name_template=""
            )
        )

        _ = (
            stats
            | "Write TFDV stats output TFRecord"
            >> tfdv.WriteStatisticsToTFRecord(self.stats_output_uri.uri)
        )

        return stats


class ReadExistingTFTransformFn(beam.PTransform):
    def __init__(self, tf_transform_directory: Uri):
        assert isinstance(tf_transform_directory, (GcsUri, LocalUri)), (
            f"tf_transform_directory must be a {GcsUri.__name__} or {LocalUri.__name__}, ",
            f"but found {tf_transform_directory.__class__.__name__}",
        )
        self.tf_transform_directory = tf_transform_directory

    def expand(self, pbegin: PBegin) -> PCollection[Any]:
        if not isinstance(pbegin, PBegin):
            raise TypeError(
                f"Input to {ReadExistingTFTransformFn.__name__} transform "
                f"must be a PBegin but found {pbegin})"
            )
        return pbegin | "Read existing TransformFn" >> tft_beam.ReadTransformFn(
            path=self.tf_transform_directory.uri
        )


class AnalyzeAndBuildTFTransformFn(beam.PTransform):
    def __init__(
        self,
        tensor_adapter_config: tfx_bsl.tfxio.tensor_adapter.TensorAdapterConfig,
        preprocessing_fn: Callable[[TFTensorDict], TFTensorDict],
    ):
        self.tensor_adapter_config = tensor_adapter_config
        self.preprocessing_fn = preprocessing_fn

    def expand(self, features: PCollection[pa.RecordBatch]) -> PCollection[Any]:
        return (
            features,
            self.tensor_adapter_config,
        ) | "Analyze raw features dataset" >> tft_beam.AnalyzeDataset(
            preprocessing_fn=self.preprocessing_fn
        )


class WriteTFSchema(beam.PTransform):
    def __init__(
        self, schema: schema_pb2.Schema, target_uri: GcsUri, schema_descriptor: str
    ):
        self.schema = schema
        self.target_uri = target_uri
        self.schema_descriptor = schema_descriptor

    def expand(self, pbegin: PBegin) -> PDone:
        if not isinstance(pbegin, PBegin):
            raise TypeError(
                f"Input to {WriteTFSchema.__name__} transform "
                f"must be a PBegin but found {pbegin})"
            )
        return (
            pbegin
            | f"Create {self.schema_descriptor} schema PCollection"
            >> beam.Create([self.schema])
            | f"Write out {self.schema_descriptor} schema proto"
            >> beam.io.WriteToText(self.target_uri.uri, shard_name_template="")
        )


def get_load_data_and_transform_pipeline_component(
    applied_task_identifier: AppliedTaskIdentifier,
    data_reference: DataReference,
    preprocessing_spec: Union[NodeDataPreprocessingSpec, EdgeDataPreprocessingSpec],
    transformed_features_info: TransformedFeaturesInfo,
    num_shards: int,
    custom_worker_image_uri: Optional[str] = None,
) -> beam.Pipeline:
    """
    Generate a Beam pipeline to conduct transformation, given a source feature table in BQ and an output path in GCS.
    """
    qualifier: str
    if isinstance(data_reference, EdgeDataReference):
        qualifier = f"-{data_reference.edge_type}-{data_reference.edge_usage_type}-"
    elif isinstance(data_reference, NodeDataReference):
        qualifier = f"-{data_reference.node_type}-"
    else:
        raise ValueError(
            f"data_reference must be of type {EdgeDataReference.__name__} or {NodeDataReference.__name__}, found: {type(data_reference)}"
        )

    job_name_suffix = f"{transformed_features_info.feature_type.value}-{qualifier}=feature-prep-{current_formatted_datetime().lower()}"
    # We disable type checking for this pipeline because it uses PTransforms with multiple PCollection inputs/outputs.
    # This is unsupported and hard to circumvent.  See https://lists.apache.org/thread/sok35vj08z8rb5drwoltkh5g06pbq19d
    # and https://lists.apache.org/thread/7cczwfz81lqrt431oh80yf3b0qwosf59.  Leaving type check enabled causes issues
    # with WriteTransformedTFRecords.

    resource_config = get_resource_config()

    if isinstance(preprocessing_spec, NodeDataPreprocessingSpec):
        data_preprocessor_config = (
            resource_config.preprocessor_config.node_preprocessor_config
        )
    elif isinstance(preprocessing_spec, EdgeDataPreprocessingSpec):
        data_preprocessor_config = (
            resource_config.preprocessor_config.edge_preprocessor_config
        )
    else:
        raise ValueError(
            f"Preprocessing spec has to be either {NodeDataPreprocessingSpec.__name__} "
            f"or {EdgeDataPreprocessingSpec.__name__}. Value given: {preprocessing_spec}"
        )

    options = init_beam_pipeline_options(
        applied_task_identifier=applied_task_identifier,
        job_name_suffix=job_name_suffix,
        component=GiGLComponents.DataPreprocessor,
        num_workers=data_preprocessor_config.num_workers,
        max_num_workers=data_preprocessor_config.max_num_workers,
        machine_type=data_preprocessor_config.machine_type,
        disk_size_gb=data_preprocessor_config.disk_size_gb,
        pipeline_type_check=False,
        resource_config=get_resource_config().get_resource_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
    )

    # pipeline start
    p = beam.Pipeline(options=options)
    with tft_beam.Context(
        temp_dir=transformed_features_info.tft_temp_directory_path.uri,
        use_deep_copy_optimization=False,
    ):
        raw_feature_spec = preprocessing_spec.feature_spec_fn()
        raw_data_schema: schema_pb2.Schema = (
            tft.tf_metadata.schema_utils.schema_from_feature_spec(raw_feature_spec)
        )

        beam_record_tfxio = tfx_bsl.tfxio.tf_example_record.TFExampleBeamRecord(
            physical_format="tfrecord", schema=raw_data_schema
        )

        raw_tensor_adapter_config = beam_record_tfxio.TensorAdapterConfig()

        # Ingest raw features from data reference and parse into TFXIO format for TFT to use.
        raw_features = p | IngestRawFeatures(
            data_reference=data_reference,
            feature_spec=raw_feature_spec,
            beam_record_tfxio=beam_record_tfxio,
        )

        # Write out the TF schema of the raw features.
        _ = p | WriteTFSchema(
            schema=raw_data_schema,
            target_uri=transformed_features_info.raw_data_schema_file_path,
            schema_descriptor="raw",
        )

        # Run TFDV and generate statistics & Facets report visualization.
        # TODO(nshah-sc): revisit commenting this out in the future as needed.
        # _ = raw_features | GenerateAndVisualizeStats(
        #     facets_report_uri=transformed_features_info.visualized_facets_file_path,
        #     stats_output_uri=transformed_features_info.stats_file_path,
        # )

        # Read previous TransformFn assets from a pretrained path if specified, else build a new asset.
        pretrained_transform_fn: Optional[Tuple[Any, Any]] = None
        analyzed_transform_fn: Optional[Tuple[Any, Any]] = None
        should_use_existing_transform_fn: bool = (
            preprocessing_spec.pretrained_tft_model_uri is not None
        )
        if should_use_existing_transform_fn:
            logger.info(
                f"Will use pretrained TFTransform asset from {preprocessing_spec.pretrained_tft_model_uri}"
            )
            assert preprocessing_spec.pretrained_tft_model_uri is not None
            pretrained_transform_fn = p | ReadExistingTFTransformFn(
                tf_transform_directory=preprocessing_spec.pretrained_tft_model_uri
            )
        else:
            logger.info(f"Will build fresh TFTransform asset.")
            analyzed_transform_fn = raw_features | AnalyzeAndBuildTFTransformFn(
                tensor_adapter_config=raw_tensor_adapter_config,
                preprocessing_fn=preprocessing_spec.preprocessing_fn,
            )

        # Write TransformFn and associated transform metadata.
        resolved_transform_fn = pretrained_transform_fn or analyzed_transform_fn
        _ = resolved_transform_fn | "Write TransformFn" >> tft_beam.WriteTransformFn(
            transformed_features_info.transform_directory_path.uri
        )

        # Write out transformed features.
        # This transformed_metadata can only be relied on for encoding purposes
        # reusing a pretrained transform_fn.  It can sometimes produce inaccurate
        # metadata for transformed features when building a new transform_fn,
        # hence we opt to use the deferred_metadata instead.
        transformed_features, transformed_metadata = (
            (raw_features, raw_tensor_adapter_config),
            resolved_transform_fn,
        ) | "Transform raw features dataset" >> tft_beam.TransformDataset()
        resolved_transformed_metadata = (
            transformed_metadata
            if should_use_existing_transform_fn
            else beam.pvalue.AsSingleton(analyzed_transform_fn[1].deferred_metadata)  # type: ignore
        )
        transformed_features | "Write tf record files" >> BetterWriteToTFRecord(
            file_path_prefix=transformed_features_info.transformed_features_file_prefix.uri,
            max_bytes_per_shard=int(2e8),  # 200mb,
            transformed_metadata=resolved_transformed_metadata,
            # TODO(mkolodner-sc): Right now, a non-zero value for num_shards overrides the max_bytes_per_shard condition. We need to implement
            # a solution where num_shards specified is just a minimum, causing the max_bytes_per_shard rule taking precedent over the num_shards rule. This will require
            # dynamically determining the number of shards produced by max_bytes_per_shard and setting it to be equal to min_num_shards if the value is less than it.
            num_shards=num_shards,
        )

        return p
