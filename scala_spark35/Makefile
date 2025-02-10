# if running outside a dataproc machine, all gs:// path should change to local paths,
# and add option --files file:/full_path_to_local_files (local files are frozen gbml, preprocessedmetadata, node, edge data)
# run under scala_spark35/
LOCAL_SPARK_JAR_PATH:=external-jars/spark_3.5.0-custom-tfrecord_2.12-0.6.1.jar
GCS_SPARK_JAR_PATH:=gs://gigl-public/tools/scala/registry/spark_3.5.0-custom-tfrecord_2.12-0.6.1.jar

LOCAL_SGS_ASSEMBLY_JAR_PATH:=subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar
GCS_SGS_ASSEMBLY_JAR_PATH:=gs://my-temp-bucket/jars/subgraph_sampler-assembly-1.0.jar

LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH:=split_generator/target/scala-2.12/split_generator-assembly-1.0.jar
GCS_SPLITGEN_ASSEMBLY_JAR_PATH:=gs://my-temp-bucket/jars/split-generator-assembly-1.0.jar

spark_run_sgs_on_current_machine:
	sbt subgraph_sampler/assembly
	../tools/scala/spark-3.5.0-bin-hadoop3/bin/spark-submit \
		--class Main \
		--master local \
		--jars $(LOCAL_SPARK_JAR_PATH) \
		$(LOCAL_SGS_ASSEMBLY_JAR_PATH) \
		common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/resource_config.yaml

spark_run_sgs_graphdb_on_current_machine:
	sbt subgraph_sampler/assembly
	../tools/scala/spark-3.5.0-bin-hadoop3/bin/spark-submit \
		--class Main \
		--master local \
		--jars $(LOCAL_SPARK_JAR_PATH) \
		$(LOCAL_SGS_ASSEMBLY_JAR_PATH) \
		common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/frozen_gbml_config_graphdb_dblp_local.yaml my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/resource_config.yaml

spark_run_splitgen_on_current_machine:
	sbt split_generator/assembly
	../tools/scala/spark-3.5.0-bin-hadoop3/bin/spark-submit \
		--class Main \
		--master local \
		--jars $(LOCAL_SPARK_JAR_PATH) \
		$(LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH) \
		my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/split_generator/hetero_node_anchor_based_link_prediction/frozen_gbml_config.yaml common/src/test/assets/resource_config.yaml

unittest:
	sbt "test:testOnly *$(test-name)"

# run `make create_test_dataproc_cluster` from root directory first
spark_run_sgs_on_cluster:
	sbt subgraph_sampler/assembly
	gsutil cp $(LOCAL_SGS_ASSEMBLY_JAR_PATH) $(GCS_SGS_ASSEMBLY_JAR_PATH)
	gcloud dataproc jobs submit spark \
	--cluster sgs-udl-mau-clus-200-8 \
	--region us-central1 \
	--jar $(GCS_SGS_ASSEMBLY_JAR_PATH)\
	--jars $(GCS_SPARK_JAR_PATH) \
	-- gs://TEST ASSET PLACEHOLDER/mocked_assets/mau_user_defined_labels/config_populator/frozen_gbml_config_sgs_225_100M.yaml my_app_name gs://DEV PERM ASSET PLACEHOLDER/resource_configs/platform/dev_mega_udl_resource_config.yaml


# create cluster beforehand with
# `python -m scripts.bootstrap_spark_cluster --use_spark35 --action="create_cluster" --component="subgraph_sampler" --cluster_name="cluster_name" --resource_config_uri resource_configs/platform/dev_mega_resource_config.yaml`
spark_run_sgs_graphdb_on_cluster:
	sbt subgraph_sampler/assembly
	gsutil cp $(LOCAL_SGS_ASSEMBLY_JAR_PATH) $(GCS_SGS_ASSEMBLY_JAR_PATH)
	gcloud dataproc jobs submit spark \
	--cluster cluster_name \
	--region us-central1 \
	--project external-snap-ci-github-gigl \
	--jar $(GCS_SGS_ASSEMBLY_JAR_PATH) \
	--jars $(GCS_SPARK_JAR_PATH) \
	-- gs://TEST ASSET PLACEHOLDER/nebula_graphdb/subgraph_sampler/configs/frozen_gbml_config_graphdb_content_cluster.yaml my_app_name gs://DEV PERM ASSET PLACEHOLDER/resource_configs/platform/dev_mega_resource_config.yaml


spark_run_splitgen_on_cluster:
	sbt split_generator/assembly
	gsutil cp $(LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH) $(GCS_SPLITGEN_ASSEMBLY_JAR_PATH)
	gcloud dataproc jobs submit spark \
	--cluster cluster_name \
	--region us-central1 \
	--jar $(GCS_SPLITGEN_ASSEMBLY_JAR_PATH) \
	--jars $(GCS_SPARK_JAR_PATH) \
	-- my_app_name gs://TEMP DEV GBML PLACEHOLDER/norway-sgs-data/config_populator/frozen_gbml_config.yaml gs://DEV PERM ASSET PLACEHOLDER/resource_configs/platform/test_default_resource_config.yaml

upload_test_assets_to_gcs:
	gsutil -m cp -r ./common/src/test/scala/assets gs://TEST ASSET PLACEHOLDER/scala
