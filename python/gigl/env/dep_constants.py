import glob
import os
from pathlib import Path

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.src.common.constants.components import GiGLComponents

# TODO: (Open Source) Make these publically accesible
GIGL_SRC_IMAGE_CUDA = (
    "gcr.io/external-snap-ci-github-gigl/gigl_src_images/gigl_src_cuda:0.0.6"
)
GIGL_SRC_IMAGE_CPU = (
    "gcr.io/external-snap-ci-github-gigl/gigl_src_images/gigl_src_cpu:0.0.6"
)
GIGL_DATAFLOW_IMAGE = (
    "gcr.io/external-snap-ci-github-gigl/gigl_src_images/gigl_src_dataflow:0.0.6"
)


_SPARK_35_DIR_NAME = "scala_spark35"
_SPARK_DIR_NAME = "scala"

GIGL_PUBLIC_BUCKET_NAME = "gigl-public"


def _get_scala_dir_name(use_spark35: bool = False) -> str:
    return _SPARK_35_DIR_NAME if use_spark35 else _SPARK_DIR_NAME


def get_compiled_jar_path(component: GiGLComponents, use_spark35=False) -> LocalUri:
    scala_dir_name = _get_scala_dir_name(use_spark35=use_spark35)
    path = LocalUri.join(
        local_fs_constants.get_project_root_directory(),
        f"{scala_dir_name}/{component.value}/target/scala-2.12/{component.value}-assembly-1.0.jar",
    )
    return path


def get_local_jar_directory(component: GiGLComponents, use_spark35=False) -> LocalUri:
    scala_dir_name = _get_scala_dir_name(use_spark35=use_spark35)
    path = LocalUri.join(
        local_fs_constants.get_gigl_root_directory(),
        f"deps/{scala_dir_name}/{component.value}/jars/",
    )
    return path


def get_current_jar_file(directory: LocalUri) -> LocalUri:
    list_of_files = glob.glob(str(Path(directory.uri) / "*.jar"))
    if not list_of_files:
        raise FileNotFoundError(f"No .jar file found in: {directory.uri}")
    latest_file = max(list_of_files, key=os.path.getctime)
    return LocalUri(latest_file)


def get_jar_file_uri(component: GiGLComponents, use_spark35=False) -> LocalUri:
    assert component in [
        GiGLComponents.SubgraphSampler,
        GiGLComponents.SplitGenerator,
    ], f"Unsupported component: {component}"
    directory = get_local_jar_directory(component=component, use_spark35=use_spark35)
    return get_current_jar_file(directory)
