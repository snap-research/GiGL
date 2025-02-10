import concurrent.futures
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import google.cloud.bigquery as bigquery
import tensorflow as tf

from gigl.common.env_config import get_available_cpus
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants import bq as bq_constants
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.enumerate import queries as enumeration_queries
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import (
    DEFAULT_TF_INT_DTYPE,
    EdgeDataPreprocessingSpec,
    FeatureSpecDict,
    NodeDataPreprocessingSpec,
)

logger = Logger()


def get_enumerated_node_id_map_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_node_{node_type}_ids_{applied_task_identifier}",
        ),
    )


def get_enumerated_node_features_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier, node_type: NodeType
) -> str:
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_node_{node_type}_node_features_{applied_task_identifier}",
        ),
    )


def get_enumerated_edge_features_bq_table_name(
    applied_task_identifier: AppliedTaskIdentifier,
    edge_type: EdgeType,
    edge_usage_type: EdgeUsageType,
):
    return BqUtils.format_bq_path(
        bq_path=BqUtils.join_path(
            bq_constants.get_embeddings_dataset_bq_path(),
            # applied_task_identifier as suffix so BQ can collapse the table names with same prefix in the UI
            f"enumerated_edge_{edge_type}_{str(edge_usage_type.value)}_edge_features_{applied_task_identifier}",
        ),
    )


def get_resource_labels() -> Dict[str, str]:
    resource_config = get_resource_config()
    return resource_config.get_resource_labels(
        component=GiGLComponents.DataPreprocessor
    )


@dataclass
class EnumeratorNodeTypeMetadata:
    input_node_data_reference: NodeDataReference
    input_node_data_preprocessing_spec: NodeDataPreprocessingSpec
    enumerated_node_data_reference: BigqueryNodeDataReference
    enumerated_node_data_preprocessing_spec: NodeDataPreprocessingSpec
    bq_unique_node_ids_enumerated_table_name: str

    def __repr__(self) -> str:
        return f"""EnumeratorNodeTypeMetadata(
            input_node_data_reference={self.input_node_data_reference},
            input_node_data_preprocessing_spec={self.input_node_data_preprocessing_spec},
            enumerated_node_data_reference={self.enumerated_node_data_reference},
            enumerated_node_data_preprocessing_spec={self.enumerated_node_data_preprocessing_spec}, bq_unique_node_ids_enumerated_table_name={self.bq_unique_node_ids_enumerated_table_name})
            """


@dataclass
class EnumeratorEdgeTypeMetadata:
    input_edge_data_reference: EdgeDataReference
    input_edge_data_preprocessing_spec: EdgeDataPreprocessingSpec
    enumerated_edge_data_reference: BigqueryEdgeDataReference
    enumerated_edge_data_preprocessing_spec: EdgeDataPreprocessingSpec

    def __repr__(self) -> str:
        return f"""EnumeratorEdgeTypeMetadata(
            input_edge_data_reference={self.input_edge_data_reference},
            input_edge_data_preprocessing_spec={self.input_edge_data_preprocessing_spec},
            enumerated_edge_data_reference={self.enumerated_edge_data_reference},
            enumerated_edge_data_preprocessing_spec={self.enumerated_edge_data_preprocessing_spec})
            """


class Enumerator:
    __applied_task_identifier: AppliedTaskIdentifier
    __bq_utils: BqUtils

    def __generate_enumerated_node_id_table_from_src_node_feature_table(
        self,
        bq_source_table_name: str,
        bq_source_table_node_id_col_name: str,
        node_type: NodeType,
    ) -> str:
        num_nodes_in_source_table = self.__bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_source_table_name, labels=get_resource_labels()
        )

        # Get unique node ids, and store to BQ.
        bq_enumerated_node_id_map_table_name: str = (
            get_enumerated_node_id_map_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                node_type=node_type,
            )
        )
        logger.info(
            f"Will write enumerated node ids to: {bq_enumerated_node_id_map_table_name}"
        )

        unique_node_enumeration_query = enumeration_queries.UNIQUE_NODE_ENUMERATION_QUERY.format(
            bq_source_table_name=bq_source_table_name,
            bq_source_table_node_id_col_name=bq_source_table_node_id_col_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            query=unique_node_enumeration_query,
            labels=get_resource_labels(),
            destination=bq_enumerated_node_id_map_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )
        num_enumerated_nodes = self.__bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_enumerated_node_id_map_table_name, labels=get_resource_labels()
        )

        # Make sure the number of input nodes and output nodes are equivalent.
        # If they are not, it suggests the input table has multiple rows for same node id.
        assert num_nodes_in_source_table == num_enumerated_nodes, (
            f"Number of input nodes not equal to number of enumerated nodes: ({num_nodes_in_source_table} != {num_enumerated_nodes}).  "
            f"Check the input table in case of duplicates."
        )
        logger.info(
            f"[Node Type: {node_type}] Finished generating enumerated ids for {num_enumerated_nodes} nodes; mapping written to {bq_enumerated_node_id_map_table_name}."
        )
        return bq_enumerated_node_id_map_table_name

    def __generate_enumerated_node_feat_table_using_node_id_map_table(
        self,
        bq_source_table_name: str,
        bq_source_table_node_id_col_name: str,
        bq_enumerated_node_id_map_table_name: str,
        node_type: NodeType,
    ) -> str:
        dst_enumerated_node_features_table_name = (
            get_enumerated_node_features_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                node_type=node_type,
            )
        )
        logger.info(
            f"[Node Type: {node_type}]: Will use enumerated node id map table: {bq_enumerated_node_id_map_table_name} to enumerate nodes in {bq_source_table_name}. Will write resulting table to {dst_enumerated_node_features_table_name}"
        )

        enumerate_node_feature_table_query = enumeration_queries.NODE_FEATURES_ENUMERATION_QUERY.format(
            bq_node_features=bq_source_table_name,
            node_id_col=bq_source_table_node_id_col_name,
            bq_enumerated_node_ids=bq_enumerated_node_id_map_table_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            enumerate_node_feature_table_query,
            labels=get_resource_config().get_resource_labels(
                component=GiGLComponents.DataPreprocessor
            ),
            destination=dst_enumerated_node_features_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )

        logger.info(
            f"[Node Type: {node_type}]: Finished writing enumerated node features to {dst_enumerated_node_features_table_name}."
        )

        return dst_enumerated_node_features_table_name

    def __enumerate_node_reference(
        self,
        node_data_ref: NodeDataReference,
        node_data_preprocessing_spec: NodeDataPreprocessingSpec,
    ) -> EnumeratorNodeTypeMetadata:
        feature_spec: FeatureSpecDict = node_data_preprocessing_spec.feature_spec_fn()
        assert (
            node_data_preprocessing_spec.identifier_output in feature_spec
        ), f"identifier_output: {node_data_preprocessing_spec.identifier_output} must be in feature_spec: {feature_spec}"

        logger.info(f"Read the following feature spec: {feature_spec}")

        if not isinstance(node_data_ref, BigqueryNodeDataReference):
            raise NotImplementedError(
                f"Enumeration currently only supported for {BigqueryNodeDataReference.__name__}"
            )
            # raw_identifier_tf_feature_type: tf.DType = raw_feature_spec[
            #     node_data_preprocessing_spec.identifier_output
            # ].dtype  # Will be used in the future; see coment below
            # TODO: (svij-sc) Support this use case by dumping data to BQ using a beam pipeline
            # Will follow up in PR

        # We expect the user to give us the actual feature spec for the node id; i.e. it might be string.
        # By the end of this function, we will finish enumerated the node id to an integer; thus we update
        # the feature spec respectively.
        feature_spec[node_data_preprocessing_spec.identifier_output] = (
            tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)
        )

        bq_source_table_name: str = BqUtils.format_bq_path(
            bq_path=node_data_ref.reference_uri,
        )
        logger.info(
            f"[Node Type: {node_data_ref.node_type}]: starting to enumerate node ids from source node table {bq_source_table_name}. The generated table will have the following feature spec: {feature_spec}"
        )
        bq_source_table_node_id_col_name: str = str(
            node_data_preprocessing_spec.identifier_output
        )
        node_type: NodeType = node_data_ref.node_type
        bq_unique_node_ids_enumerated_table_name: str = (
            self.__generate_enumerated_node_id_table_from_src_node_feature_table(
                bq_source_table_name=bq_source_table_name,
                bq_source_table_node_id_col_name=bq_source_table_node_id_col_name,
                node_type=node_type,
            )
        )
        bq_destination_enumerated_node_features_table_name: str = (
            self.__generate_enumerated_node_feat_table_using_node_id_map_table(
                bq_source_table_name=bq_source_table_name,
                bq_source_table_node_id_col_name=bq_source_table_node_id_col_name,
                bq_enumerated_node_id_map_table_name=bq_unique_node_ids_enumerated_table_name,
                node_type=node_type,
            )
        )

        enumerated_node_data_preprocessing_spec = NodeDataPreprocessingSpec(
            feature_spec_fn=lambda: feature_spec,
            preprocessing_fn=node_data_preprocessing_spec.preprocessing_fn,
            identifier_output=node_data_preprocessing_spec.identifier_output,
            pretrained_tft_model_uri=node_data_preprocessing_spec.pretrained_tft_model_uri,
            features_outputs=node_data_preprocessing_spec.features_outputs,
            labels_outputs=node_data_preprocessing_spec.labels_outputs,
        )

        return EnumeratorNodeTypeMetadata(
            input_node_data_reference=node_data_ref,
            input_node_data_preprocessing_spec=node_data_preprocessing_spec,
            enumerated_node_data_reference=BigqueryNodeDataReference(
                reference_uri=bq_destination_enumerated_node_features_table_name,
                node_type=node_type,
            ),
            enumerated_node_data_preprocessing_spec=enumerated_node_data_preprocessing_spec,
            bq_unique_node_ids_enumerated_table_name=bq_unique_node_ids_enumerated_table_name,
        )

    def __enumerate_all_node_references(
        self,
        node_preprocessing_specs: Dict[NodeDataReference, NodeDataPreprocessingSpec],
    ) -> List[EnumeratorNodeTypeMetadata]:
        results: List[EnumeratorNodeTypeMetadata] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=get_available_cpus()
        ) as executor:
            futures: List[concurrent.futures.Future] = list()
            for (
                node_data_ref,
                node_data_preprocessing_spec,
            ) in node_preprocessing_specs.items():
                future = executor.submit(
                    self.__enumerate_node_reference,
                    node_data_ref=node_data_ref,
                    node_data_preprocessing_spec=node_data_preprocessing_spec,
                )
                futures.append(future)

            for future in futures:
                result: EnumeratorNodeTypeMetadata = future.result()
                results.append(result)

        return results

    def __enumerate_all_edge_references(
        self,
        edge_preprocessing_specs: Dict[EdgeDataReference, EdgeDataPreprocessingSpec],
        map_enumerator_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata],
    ) -> List[EnumeratorEdgeTypeMetadata]:
        results: List[EnumeratorEdgeTypeMetadata] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=get_available_cpus()
        ) as executor:
            futures: List[concurrent.futures.Future] = list()
            for (
                edge_data_ref,
                edge_preprocessing_spec,
            ) in edge_preprocessing_specs.items():
                future = executor.submit(
                    self.__enumerate_edge_reference,
                    edge_data_ref=edge_data_ref,
                    edge_preprocessing_spec=edge_preprocessing_spec,
                    map_enumerator_node_type_metadata=map_enumerator_node_type_metadata,
                )
                futures.append(future)

            for future in futures:
                result: EnumeratorEdgeTypeMetadata = future.result()
                results.append(result)

        return results

    def __generate_enumerated_edge_feat_table_using_node_id_map_tables(
        self,
        edge_type: EdgeType,
        edge_usage_type: EdgeUsageType,
        bq_source_table_name: str,
        bq_source_table_src_node_id_col_name: str,
        bq_source_table_dst_node_id_col_name: str,
        bq_enumerated_src_node_id_map_table_name: str,
        bq_enumerated_dst_node_id_map_table_name: str,
        has_edge_features: bool,
    ) -> str:
        dst_enumerated_edge_features_table_name: str = (
            get_enumerated_edge_features_bq_table_name(
                applied_task_identifier=self.__applied_task_identifier,
                edge_type=edge_type,
                edge_usage_type=edge_usage_type,
            )
        )
        graph_edges_enumeration_query = (
            enumeration_queries.EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY
            if has_edge_features
            else enumeration_queries.NO_EDGE_FEATURES_GRAPH_EDGELIST_ENUMERATION_QUERY
        ).format(
            bq_graph=bq_source_table_name,
            src_enumerated_node_ids=bq_enumerated_src_node_id_map_table_name,
            dst_enumerated_node_ids=bq_enumerated_dst_node_id_map_table_name,
            src_node_id_col=bq_source_table_src_node_id_col_name,
            dst_node_id_col=bq_source_table_dst_node_id_col_name,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        )

        self.__bq_utils.run_query(
            query=graph_edges_enumeration_query,
            labels=get_resource_config().get_resource_labels(
                component=GiGLComponents.DataPreprocessor
            ),
            destination=dst_enumerated_edge_features_table_name,
            write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
        )
        return dst_enumerated_edge_features_table_name

    def __enumerate_edge_reference(
        self,
        edge_data_ref: EdgeDataReference,
        edge_preprocessing_spec: EdgeDataPreprocessingSpec,
        map_enumerator_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata],
    ) -> EnumeratorEdgeTypeMetadata:
        feature_spec: FeatureSpecDict = edge_preprocessing_spec.feature_spec_fn()
        assert (
            edge_preprocessing_spec.identifier_output.src_node in feature_spec
        ), f"identifier_output: {edge_preprocessing_spec.identifier_output.src_node} must be in feature_spec: {feature_spec}"
        assert (
            edge_preprocessing_spec.identifier_output.dst_node in feature_spec
        ), f"identifier_output: {edge_preprocessing_spec.identifier_output.dst_node} must be in feature_spec: {feature_spec}"
        logger.info(f"Read the following feature spec: {feature_spec}")

        if not isinstance(edge_data_ref, BigqueryEdgeDataReference):
            raise NotImplementedError(
                f"Enumeration currently only supported for {BigqueryEdgeDataReference.__name__}"
            )
            # TODO: (svij-sc) Support this use case by dumping data to BQ using a beam pipeline
            # Will follow up in PR

        # We expect the user to give us the actual feature spec for the node id; i.e. it might be string.
        # By the end of this function, we will finish enumerated the node ids for edges to integers;
        # thus we update the feature spec respectively.
        feature_spec[edge_preprocessing_spec.identifier_output.src_node] = (
            tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)
        )
        feature_spec[edge_preprocessing_spec.identifier_output.dst_node] = (
            tf.io.FixedLenFeature(shape=[], dtype=DEFAULT_TF_INT_DTYPE)
        )

        bq_source_table_name: str = BqUtils.format_bq_path(
            bq_path=edge_data_ref.reference_uri,
        )
        bq_source_table_src_node_id_col_name: str = str(
            edge_preprocessing_spec.identifier_output.src_node
        )
        bq_source_table_dst_node_id_col_name: str = str(
            edge_preprocessing_spec.identifier_output.dst_node
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: starting to enumerate node ids from source edge table {bq_source_table_name}. The generated table will have the following feature spec: {feature_spec}"
        )

        # Get source and destination metadata.
        src_node_type, dst_node_type = (
            edge_data_ref.edge_type.src_node_type,
            edge_data_ref.edge_type.dst_node_type,
        )
        src_enumerated_node_type_metadata = map_enumerator_node_type_metadata[
            src_node_type
        ]
        dst_enumerated_node_type_metadata = map_enumerator_node_type_metadata[
            dst_node_type
        ]

        src_enumerated_node_ids = BqUtils.format_bq_path(
            bq_path=src_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name
        )
        dst_enumerated_node_ids = BqUtils.format_bq_path(
            bq_path=dst_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name
        )

        has_edge_features: bool = (
            edge_preprocessing_spec.features_outputs is not None
            and len(edge_preprocessing_spec.features_outputs) > 0
        ) or (
            edge_preprocessing_spec.labels_outputs is not None
            and len(edge_preprocessing_spec.labels_outputs) > 0
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: Started writing enumerated edges (and features)."
        )

        bq_enumerated_edge_features_table_name = self.__generate_enumerated_edge_feat_table_using_node_id_map_tables(
            edge_type=edge_data_ref.edge_type,
            edge_usage_type=edge_data_ref.edge_usage_type,
            bq_source_table_name=bq_source_table_name,
            bq_source_table_src_node_id_col_name=bq_source_table_src_node_id_col_name,
            bq_source_table_dst_node_id_col_name=bq_source_table_dst_node_id_col_name,
            bq_enumerated_src_node_id_map_table_name=src_enumerated_node_ids,
            bq_enumerated_dst_node_id_map_table_name=dst_enumerated_node_ids,
            has_edge_features=has_edge_features,
        )

        logger.info(
            f"[Edge Type: {edge_data_ref.edge_type} ; Edge Classification: {edge_data_ref.edge_usage_type}]: Finished writing enumerated edges (and features) to {bq_enumerated_edge_features_table_name}."
        )

        enumerated_edge_data_preprocessing_spec = EdgeDataPreprocessingSpec(
            feature_spec_fn=lambda: feature_spec,
            preprocessing_fn=edge_preprocessing_spec.preprocessing_fn,
            identifier_output=edge_preprocessing_spec.identifier_output,
            pretrained_tft_model_uri=edge_preprocessing_spec.pretrained_tft_model_uri,
            features_outputs=edge_preprocessing_spec.features_outputs,
            labels_outputs=edge_preprocessing_spec.labels_outputs,
        )

        return EnumeratorEdgeTypeMetadata(
            input_edge_data_reference=edge_data_ref,
            input_edge_data_preprocessing_spec=edge_preprocessing_spec,
            enumerated_edge_data_reference=BigqueryEdgeDataReference(
                reference_uri=bq_enumerated_edge_features_table_name,
                edge_type=edge_data_ref.edge_type,
                edge_usage_type=edge_data_ref.edge_usage_type,
            ),
            enumerated_edge_data_preprocessing_spec=enumerated_edge_data_preprocessing_spec,
        )

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        node_preprocessing_specs: Dict[NodeDataReference, NodeDataPreprocessingSpec],
        edge_preprocessing_specs: Dict[EdgeDataReference, EdgeDataPreprocessingSpec],
        gcp_project: str,
    ) -> Tuple[List[EnumeratorNodeTypeMetadata], List[EnumeratorEdgeTypeMetadata]]:
        self.__bq_utils = BqUtils(project=gcp_project)
        self.__applied_task_identifier = applied_task_identifier

        enumerated_node_metadata: List[EnumeratorNodeTypeMetadata] = (
            self.__enumerate_all_node_references(
                node_preprocessing_specs=node_preprocessing_specs
            )
        )
        map_enumerator_node_type_metadata: Dict[
            NodeType, EnumeratorNodeTypeMetadata
        ] = {
            node_metadata.input_node_data_reference.node_type: node_metadata
            for node_metadata in enumerated_node_metadata
        }
        enumerated_edge_metadata: List[EnumeratorEdgeTypeMetadata] = (
            self.__enumerate_all_edge_references(
                edge_preprocessing_specs=edge_preprocessing_specs,
                map_enumerator_node_type_metadata=map_enumerator_node_type_metadata,
            )
        )

        logger.info("Finished enumerating all node and edge references.")
        logger.info("Generated the following node enumerations:")
        for node_metadata in enumerated_node_metadata:
            logger.info(node_metadata)
        logger.info("Generated the following edge enumerations:")
        for edge_metadata in enumerated_edge_metadata:
            logger.info(edge_metadata)

        return (enumerated_node_metadata, enumerated_edge_metadata)

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        node_preprocessing_specs: Dict[NodeDataReference, NodeDataPreprocessingSpec],
        edge_preprocessing_specs: Dict[EdgeDataReference, EdgeDataPreprocessingSpec],
        gcp_project: str,
    ) -> Tuple[List[EnumeratorNodeTypeMetadata], List[EnumeratorEdgeTypeMetadata]]:
        try:
            return self.__run(
                applied_task_identifier=applied_task_identifier,
                node_preprocessing_specs=node_preprocessing_specs,
                edge_preprocessing_specs=edge_preprocessing_specs,
                gcp_project=gcp_project,
            )
        except Exception as e:
            logger.error(
                "Enumerator failed due to a raised exception, which will follow"
            )
            logger.error(e)
            logger.error(traceback.format_exc())
            sys.exit(f"System will now exit: {e}")
