# Config Populator

The Config Populator takes a "template" config and generates a "frozen" config to be used by all subsequent components.

##  Input
- **job_name** (AppliedTaskIdentifier):  which uniquely identifies an end-to-end task.
- **task_config_uri** (Uri):  Path which points to a "template" `GbmlConfig` proto yaml file.
- **resource_config_uri** (Uri): Path which points to a `GiGLResourceConfig` yaml

##  What does it do?
Takes in a template `GbmlConfig` and outputs a frozen `GbmlConfig` by populating all job related metadata paths in `sharedConfig`. These are mostly GCS paths which the following components read and write from, and use as an intermediary data communication medium. For example, the field `sharedConfig.trainedModelMetadata` is populated with a GCS URI, which indicates to the Trainer to write the trained model to this path, and to the Inferencer to read the model from this path

## How do I run it?

**Import GiGL**

```python
from gigl.src.config_populator.config_populator import ConfigPopulator
from gigl.common import UriFactory
from gigl.src.common.types import AppliedTaskIdentifier

config_populator = ConfigPopulator()

task_config_uri = config_populator.run(
    applied_task_identifier=AppliedTaskIdentifier("my_gigl_job_name"),
    task_config_uri=UriFactory.create_uri("gs://my-temp-assets-bucket/task_config.yaml"),
    resource_config_uri=UriFactory.create_uri("gs://my-temp-assets-bucket/resource_config.yaml")
)
```

**Command Line**
```bash
python -m \
    gigl.src.config_populator.config_populator \
    --job_name mau_sample_task_id \
    --template_uri gs://TEST ASSET PLACEHOLDER/example_task_assets/mau/configs/mau_template_gbml_config_20230213.yaml \
    --output_file_path_frozen_gbml_config_uri my-output-frozen-config_path.yaml
```

**Notes:**

- `output_file_path_frozen_gbml_config_uri` is the output of the run method as seen in the import gigl usage. 
- Be sure to note the discrepency of `template_uri` for command line usage vs `task_config_uri` for import gigl usage. 

## Output

A frozen `GbmlConfig` URI. 

## Other

- If `trainedModelMetadata.trainedModelUri` exists and/or `skipTraining = true`, this indicates that we will be running the pipeline with a pre-trained model, and Config Populator will not overwrite these fields in the `sharedConfig` of the frozen `GbmlConfig`.

- Although `sharedConfig` is added to the frozen config by the config populator, you may add the field to your template config to enable any feature/optional flags. 