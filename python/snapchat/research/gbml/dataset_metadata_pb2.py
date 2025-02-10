# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: snapchat/research/gbml/dataset_metadata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-snapchat/research/gbml/dataset_metadata.proto\x12\x16snapchat.research.gbml\"j\n#SupervisedNodeClassificationDataset\x12\x16\n\x0etrain_data_uri\x18\x01 \x01(\t\x12\x15\n\rtest_data_uri\x18\x02 \x01(\t\x12\x14\n\x0cval_data_uri\x18\x03 \x01(\t\"\xb2\x06\n$NodeAnchorBasedLinkPredictionDataset\x12\x1b\n\x13train_main_data_uri\x18\x01 \x01(\t\x12\x1a\n\x12test_main_data_uri\x18\x02 \x01(\t\x12\x19\n\x11val_main_data_uri\x18\x03 \x01(\t\x12\x9b\x01\n+train_node_type_to_random_negative_data_uri\x18\x04 \x03(\x0b\x32\x66.snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.TrainNodeTypeToRandomNegativeDataUriEntry\x12\x97\x01\n)val_node_type_to_random_negative_data_uri\x18\x05 \x03(\x0b\x32\x64.snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.ValNodeTypeToRandomNegativeDataUriEntry\x12\x99\x01\n*test_node_type_to_random_negative_data_uri\x18\x06 \x03(\x0b\x32\x65.snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.TestNodeTypeToRandomNegativeDataUriEntry\x1aK\n)TrainNodeTypeToRandomNegativeDataUriEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1aI\n\'ValNodeTypeToRandomNegativeDataUriEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1aJ\n(TestNodeTypeToRandomNegativeDataUriEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"j\n#SupervisedLinkBasedTaskSplitDataset\x12\x16\n\x0etrain_data_uri\x18\x01 \x01(\t\x12\x15\n\rtest_data_uri\x18\x02 \x01(\t\x12\x14\n\x0cval_data_uri\x18\x03 \x01(\t\"\xf1\x02\n\x0f\x44\x61tasetMetadata\x12m\n&supervised_node_classification_dataset\x18\x01 \x01(\x0b\x32;.snapchat.research.gbml.SupervisedNodeClassificationDatasetH\x00\x12q\n)node_anchor_based_link_prediction_dataset\x18\x02 \x01(\x0b\x32<.snapchat.research.gbml.NodeAnchorBasedLinkPredictionDatasetH\x00\x12i\n\"supervised_link_based_task_dataset\x18\x03 \x01(\x0b\x32;.snapchat.research.gbml.SupervisedLinkBasedTaskSplitDatasetH\x00\x42\x11\n\x0foutput_metadatab\x06proto3')



_SUPERVISEDNODECLASSIFICATIONDATASET = DESCRIPTOR.message_types_by_name['SupervisedNodeClassificationDataset']
_NODEANCHORBASEDLINKPREDICTIONDATASET = DESCRIPTOR.message_types_by_name['NodeAnchorBasedLinkPredictionDataset']
_NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY = _NODEANCHORBASEDLINKPREDICTIONDATASET.nested_types_by_name['TrainNodeTypeToRandomNegativeDataUriEntry']
_NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY = _NODEANCHORBASEDLINKPREDICTIONDATASET.nested_types_by_name['ValNodeTypeToRandomNegativeDataUriEntry']
_NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY = _NODEANCHORBASEDLINKPREDICTIONDATASET.nested_types_by_name['TestNodeTypeToRandomNegativeDataUriEntry']
_SUPERVISEDLINKBASEDTASKSPLITDATASET = DESCRIPTOR.message_types_by_name['SupervisedLinkBasedTaskSplitDataset']
_DATASETMETADATA = DESCRIPTOR.message_types_by_name['DatasetMetadata']
SupervisedNodeClassificationDataset = _reflection.GeneratedProtocolMessageType('SupervisedNodeClassificationDataset', (_message.Message,), {
  'DESCRIPTOR' : _SUPERVISEDNODECLASSIFICATIONDATASET,
  '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
  # @@protoc_insertion_point(class_scope:snapchat.research.gbml.SupervisedNodeClassificationDataset)
  })
_sym_db.RegisterMessage(SupervisedNodeClassificationDataset)

NodeAnchorBasedLinkPredictionDataset = _reflection.GeneratedProtocolMessageType('NodeAnchorBasedLinkPredictionDataset', (_message.Message,), {

  'TrainNodeTypeToRandomNegativeDataUriEntry' : _reflection.GeneratedProtocolMessageType('TrainNodeTypeToRandomNegativeDataUriEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY,
    '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
    # @@protoc_insertion_point(class_scope:snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.TrainNodeTypeToRandomNegativeDataUriEntry)
    })
  ,

  'ValNodeTypeToRandomNegativeDataUriEntry' : _reflection.GeneratedProtocolMessageType('ValNodeTypeToRandomNegativeDataUriEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY,
    '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
    # @@protoc_insertion_point(class_scope:snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.ValNodeTypeToRandomNegativeDataUriEntry)
    })
  ,

  'TestNodeTypeToRandomNegativeDataUriEntry' : _reflection.GeneratedProtocolMessageType('TestNodeTypeToRandomNegativeDataUriEntry', (_message.Message,), {
    'DESCRIPTOR' : _NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY,
    '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
    # @@protoc_insertion_point(class_scope:snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset.TestNodeTypeToRandomNegativeDataUriEntry)
    })
  ,
  'DESCRIPTOR' : _NODEANCHORBASEDLINKPREDICTIONDATASET,
  '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
  # @@protoc_insertion_point(class_scope:snapchat.research.gbml.NodeAnchorBasedLinkPredictionDataset)
  })
_sym_db.RegisterMessage(NodeAnchorBasedLinkPredictionDataset)
_sym_db.RegisterMessage(NodeAnchorBasedLinkPredictionDataset.TrainNodeTypeToRandomNegativeDataUriEntry)
_sym_db.RegisterMessage(NodeAnchorBasedLinkPredictionDataset.ValNodeTypeToRandomNegativeDataUriEntry)
_sym_db.RegisterMessage(NodeAnchorBasedLinkPredictionDataset.TestNodeTypeToRandomNegativeDataUriEntry)

SupervisedLinkBasedTaskSplitDataset = _reflection.GeneratedProtocolMessageType('SupervisedLinkBasedTaskSplitDataset', (_message.Message,), {
  'DESCRIPTOR' : _SUPERVISEDLINKBASEDTASKSPLITDATASET,
  '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
  # @@protoc_insertion_point(class_scope:snapchat.research.gbml.SupervisedLinkBasedTaskSplitDataset)
  })
_sym_db.RegisterMessage(SupervisedLinkBasedTaskSplitDataset)

DatasetMetadata = _reflection.GeneratedProtocolMessageType('DatasetMetadata', (_message.Message,), {
  'DESCRIPTOR' : _DATASETMETADATA,
  '__module__' : 'snapchat.research.gbml.dataset_metadata_pb2'
  # @@protoc_insertion_point(class_scope:snapchat.research.gbml.DatasetMetadata)
  })
_sym_db.RegisterMessage(DatasetMetadata)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY._options = None
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_options = b'8\001'
  _NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY._options = None
  _NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_options = b'8\001'
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY._options = None
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_options = b'8\001'
  _SUPERVISEDNODECLASSIFICATIONDATASET._serialized_start=73
  _SUPERVISEDNODECLASSIFICATIONDATASET._serialized_end=179
  _NODEANCHORBASEDLINKPREDICTIONDATASET._serialized_start=182
  _NODEANCHORBASEDLINKPREDICTIONDATASET._serialized_end=1000
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_start=774
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TRAINNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_end=849
  _NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_start=851
  _NODEANCHORBASEDLINKPREDICTIONDATASET_VALNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_end=924
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_start=926
  _NODEANCHORBASEDLINKPREDICTIONDATASET_TESTNODETYPETORANDOMNEGATIVEDATAURIENTRY._serialized_end=1000
  _SUPERVISEDLINKBASEDTASKSPLITDATASET._serialized_start=1002
  _SUPERVISEDLINKBASEDTASKSPLITDATASET._serialized_end=1108
  _DATASETMETADATA._serialized_start=1111
  _DATASETMETADATA._serialized_end=1480
# @@protoc_insertion_point(module_scope)
