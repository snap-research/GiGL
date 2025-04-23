#!/bin/bash
set -e
set -x

# Configure docker for gcloud and multi-arch builds.
# Needed for `make push_dataflow_docker_image`.
docker buildx create --driver=docker-container --use

gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
gcloud auth list
gcloud config list project
