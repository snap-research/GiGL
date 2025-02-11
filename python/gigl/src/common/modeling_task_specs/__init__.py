# autoflake: skip_file
"""
Task specs for training / inferring on models.
- All trainer implementations should subclass gigl.src.training.v1.lib.base_trainer.BaseTrainer
- All inferencer implementations should subclass from gigl.src.inference.v1.lib.base_inferencer.BaseInferencer
"""

from gigl.src.common.modeling_task_specs.node_anchor_based_link_prediction_modeling_task_spec import NodeAnchorBasedLinkPredictionModelingTaskSpec
from gigl.src.common.modeling_task_specs.node_classification_modeling_task_spec import NodeClassificationModelingTaskSpec
from gigl.src.common.modeling_task_specs.graphsage_template_modeling_spec import GraphSageTemplateTrainerSpec
