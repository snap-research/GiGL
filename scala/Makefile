# if running outside a dataproc machine, all gs:// path should change to local paths,
# and add option --files file:/full_path_to_local_files (local files are frozen gbml, preprocessedmetadata, node, edge data)
# run under scala/
spark_run_sgs_on_current_machine:
	sbt subgraph_sampler/assembly
	../tools/scala/spark-3.1.3-bin-hadoop3.2/bin/spark-submit \
		--class Main \
		--master local \
		--jars external-jars/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
		subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar \
		common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/resource_config.yaml

# TODO: (yliu2) Remove after migration to graph service?
spark_run_sgs_graphdb_on_current_machine:
	sbt subgraph_sampler/assembly
	../tools/scala/spark-3.1.3-bin-hadoop3.2/bin/spark-submit \
		--class Main \
		--master local \
		--jars external-jars/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
		subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar \
		common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/frozen_gbml_config_graphdb_dblp_local.yaml my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/resource_config.yaml

spark_run_splitgen_on_current_machine:
	sbt split_generator/assembly
	../tools/scala/spark-3.1.3-bin-hadoop3.2/bin/spark-submit \
		--class Main \
		--master local \
		--jars external-jars/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
		split_generator/target/scala-2.12/split_generator-assembly-1.0.jar \
		my_job_name_which_will_be_the_sparkapp_name common/src/test/assets/split_generator/node_anchor_based_link_prediction/frozen_gbml_config.yaml common/src/test/assets/resource_config.yaml

unittest:
	sbt "test:testOnly *$(test-name)"

# run `make create_test_dataproc_cluster` from root directory first
spark_run_sgs_on_cluster:
	sbt subgraph_sampler/assembly
	gsutil cp subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/subgraph_sampler-assembly-1.0.jar
	gcloud dataproc jobs submit spark \
	--cluster sgs-udl-mau-clus-200-8 \
	--region us-central1 \
	--jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/subgraph_sampler-assembly-1.0.jar \
	--jars gs://public-gigl/tools/scala/spark_packages/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
	-- gs://TEST ASSET PLACEHOLDER/mocked_assets/mau_user_defined_labels/config_populator/frozen_gbml_config_sgs_225_100M.yaml my_app_name gs://DEV PERM ASSET PLACEHOLDER/resource_configs/platform/dev_mega_udl_resource_config.yaml


# create cluster beforehand with
# `python -m scripts.bootstrap_spark_cluster --action="create_cluster" --component="subgraph_sampler" --cluster_name="neb-content2" --resource_config_uri internal/resource_configs/platform/dev_mega_resource_config.yaml`
spark_run_sgs_graphdb_on_cluster:
	sbt subgraph_sampler/assembly
	gsutil cp subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/subgraph_sampler-assembly-1.0.jar
	gcloud dataproc jobs submit spark \
	--cluster neb-content2 \
	--region us-central1 \
	--project external-snap-ci-github-gigl \
	--jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/subgraph_sampler-assembly-1.0.jar \
	--jars gs://public-gigl/tools/scala/spark_packages/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
	-- gs://TEST ASSET PLACEHOLDER/nebula_graphdb/subgraph_sampler/configs/frozen_gbml_config_graphdb_content_cluster.yaml my_app_name gs://DEV PERM ASSET PLACEHOLDER/resource_configs/platform/dev_mega_resource_config.yaml
+

spark_run_splitgen_on_cluster:
	sbt split_generator/assembly
	gsutil cp split_generator/target/scala-2.12/split_generator-assembly-1.0.jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/split_generator-assembly-1.0.jar
	gcloud dataproc jobs submit spark \
	--cluster refactor \
	--region us-central1 \
	--jar gs://TEMP DEV GBML PLACEHOLDER/elham_test/split_generator-assembly-1.0.jar \
	--jars gs://public-gigl/tools/scala/spark_packages/snap-spark-custom-tfrecord_2.12-0.5.0.jar \
	-- my_app_name gs://TEMP DEV GBML PLACEHOLDER/norway-sgs-data/config_populator/frozen_gbml_config.yaml deployment/configs/unittest_resource_config.yaml

upload_test_assets_to_gcs:
	gsutil -m cp -r ./common/src/test/scala/assets gs://TEST ASSET PLACEHOLDER/scala

