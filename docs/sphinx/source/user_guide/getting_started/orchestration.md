# Orchestration

GiGL is designed to support easy end to end orchestration of your GNN tasks/workflows with minimal setup required. This page outlines three ways to orchestrate GiGL for after you have set up your configs (See: [quick start](../config_guides/task_config_guide.md) if you have not done so). 

## Local Runner
The local runner provides a simple interface to kick off an end to end GiGL pipeline. 

1. Create a pipeline config. The pipeline config takes in:
- applied_task_identifier: your job name (string)
- template_task_config_uri: the URI to your template task config (Uri)
- resource_config_uri: The URI to your resource config (URI)
- custom_cuda_docker_uri: For custom training spec and GPU training on VertexAI (optional, string)
- custom_cpu_docker_uri: For custom training spec and CPU training on VertexAI (optional, string)
- dataflow_docker_uri: For custom datapreprocessor spec that will run in dataflow (optional, string)

Example:

```python
from gigl.orchestration.local.runner import PipelineConfig
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.common import UriFactory, Uri

PipelineConfig(
    applied_task_identifier=AppliedTaskIdentifier("demo-gigl-job"),
    template_task_config_uri=UriFactory.create_uri("gs://project/my_task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://project/my_resource_config.yaml")
)
```

2. Initialize and Run

Now you can create the GiGL runner object and kick off a pipeline. 

Example:

```python

runner = Runner.run(pipeline_config=pipeline_config)

```

3. Optional: The runner also supports running individual components as needed

Example:

```python
Runner.run_data_preprocessor(pipeline_config=pipeline_config)
```

## Kubeflow Orchestration

GiGL also supports orchestration of your workflows using [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v2/) using a `KfpOrchestrator` class.

This guide assumes you have a KFP pipeline deployed and access to the experiment ID, pipeline ID, and KFP host. If not, you can see [Kubeflow Docs](https://www.kubeflow.org/docs/components/pipelines/v1/sdk/connect-api/).

The `KfpOrchestrator` requires your Kubeflow metadata, which can be passed in 2 different ways. 

- **`kfp_metadata`** (`KfpEnvMetadata`, optional): Instance containing KFP environment metadata. Loaded from environment variables if not provided.
- **`env_path`** (`str`, optional): Path to the environment file with KFP metadata. Defaults to the current directory.


**KFP_HOST**: This is the URL of your Kubeflow Pipelines endpoint. You can find it in the Kubeflow user interface or in your cloud provider's Kubernetes cluster configuration where Kubeflow is deployed.

**K8_SA** (Kubernetes Service Account): The name of the Kubernetes service account used by your pipeline runs. This can typically be found in your Kubernetes cluster configuration or set up through your cluster's service account management.

**EXPERIMENT_ID**: This is the ID of the experiment under which your pipeline runs will be grouped. You can create a new experiment in the Kubeflow Pipelines UI and use the generated ID, or you can use the ID of an existing experiment.

**PIPELINE_ID**: The unique identifier for your pipeline. This ID can be obtained from the Kubeflow Pipelines UI after you have initially uploaded your pipeline.

**EXPERIMENT_NAME** (optional): The name of the experiment in Kubeflow. This is usually set when you create an experiment in the Kubeflow UI and can be used for easier identification of your experiment runs.

For more information see [Kubeflow Quickstart](https://www.kubeflow.org/docs/components/pipelines/v2/installation/quickstart/)

The .env file would look like:

```
KFP_HOST=https://example-kubeflow-host.com
K8_SA=my-k8-service-account
EXPERIMENT_ID=my-experiment-id
PIPELINE_ID=my-pipeline-id
EXPERIMENT_NAME=my-experiment-name
```

### Usage Example

```python

from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

orchestrator = KfpOrchestrator("path/to/.env")

task_config_uri = UriFactory.create_uri("gs://path/to/task_config.yaml")
resource_config_uri = UriFactory.create_uri("gs://path/to/resource_config.yaml")
applied_task_identifier = AppliedTaskIdentifier("kfp_demo")

orchestrator.run(
    applied_task_identifier=applied_task_identifier,
    task_config_uri=task_config_uri,
    resource_config_uri=resource_config_uri,
    start_at="config_populator"
)

```


## Importable GiGL

You may want to integrate gigl into your existing workflows or create custom orchestration logic. This can be done by importing the components and using each of their `.run()` components. 

### Trainer Component Example:

```python
from gigl.src.training.trainer import Trainer

trainer = Trainer()
trainer.run(
    applied_task_identifier=job_name,
    task_config_uri="gs://...",
    resource_config_uri=="gs://...",
)
```

For component specific parameters/information, see [Components](../overview/components.md)
