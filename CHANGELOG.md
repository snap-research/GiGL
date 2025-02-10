# GiGL Changelog

All notable changes to this repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Latest Release: 0.0.6

## [Unreleased]

### Added

- Introduce BaseIngestion and GraphDBConfig (#115)(#116)
- Add retry logic when creating dataproc cluster (#124)
- Introduce `should_include_isolated_nodes_in_training` field in shared config (#134)

### Changed

- Config Validation now also checks GiglResourceConfig and PreprocessedMetadata (#123)(#125)
- Simplified Dockerfiles to be 40-50\% smaller (#198)

### Deprecated

### Removed

### Fixed

- Corrected invalid transductive node classification test (#128) 
- Corrected some inconsistencies in component cleanup logic (#196)

## [0.0.6] - 2024-05-16

### Added

- Introduce TaskMetadata and GraphMetadata in GbmlConfig (#830)
- Sphinx Support (#19)
- `stop_after` parameter in KFP pipeline (#49)
- UDLAnchorBasedSupervisionEdge Split Strategy (#17)
- Added build and push docker image script (#109)
- SortedDict Class for unsorted proto fields used in HGS (#150)
- Add SimpleHGN model for HGS (#122)

### Changed

- Logger Refactor (#29)
- Unsupervised Node Anchor (UNA) -> Node Anchor (NA) Refactor (#40)
- GiGL Modeling Task Spec Infer/Forward Refactor (#43)
- UNALP Batch + Sample Definitions (#32)
- RNN Samples paths and introduce GbmlConfigPbWrapper (#10)
- Complex Modeling Task Spec - Testing (#61)
- Complex Modeling Task Spec - Task Spec(#60)
- Complex Modeling Task Spec - Task Initialization Changes (#71)
- Complex Modeling Task Spec - Utility Functions for Training (#57, #62)
- HGS Support for Trainer (#89)
- HGS Support for Inferencer - Proto Changes (#121)
- HGS Support for Inferencer - Codebase Changes (#113)
- HGS Support for Trainer/Inferencer - Mocking/Testing Changes (#110)

### Deprecated

- Deprecation of `UserDefLabelType`; consolidation into `EdgeUsageType` (#18)
- Deprecate cora_assets.py and toy_graph.py in favour of `PassthroughPreprocessorConfigForMockedAssets` (#25)

### Fixed
- Make feature order determinisitc in FeatureEmbeddingLayer (#23)

## [0.0.5] - 2024

### Added

- Introduce KFPOrchestrator
- Enable CPU training
- Cost Tooling Script
- Torch, TFT, TF, PyG upgrades + mac arm64 support (#807)


### Fixes

- None