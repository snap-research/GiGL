from pathlib import Path
from typing import Dict, List

GIGL_ROOT_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
)  # common -> gigl -> python -> root
PATH_GIGL_PKG_INIT_FILE: Path = Path.joinpath(
    GIGL_ROOT_DIR, "python", "gigl", "__init__.py"
)
PATH_BASE_IMAGES_VARIABLE_FILE: Path = Path.joinpath(
    GIGL_ROOT_DIR, "dep_vars.env"
).absolute()

def parse_makefile_vars(makefile_path: Path) -> Dict[str, str]:
    vars_dict: Dict[str, str] = {}
    lines: List[str] = []
    with open(makefile_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            continue
        if "=" in line:
            key, value = line.split("=")
            vars_dict[key.strip()] = value.strip()
    return vars_dict


_make_file_vars: Dict[str, str] = parse_makefile_vars(PATH_BASE_IMAGES_VARIABLE_FILE)

DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG: str = _make_file_vars[
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG: str = _make_file_vars[
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG: str = _make_file_vars[
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG"
]
SPARK_35_TFRECORD_JAR_GCS_PATH: str = _make_file_vars["SPARK_35_TFRECORD_JAR_GCS_PATH"]
SPARK_31_TFRECORD_JAR_GCS_PATH: str = _make_file_vars["SPARK_31_TFRECORD_JAR_GCS_PATH"]

# Ensure that the local path is a fully resolved local path
SPARK_35_TFRECORD_JAR_LOCAL_PATH: str = str(
    Path.joinpath(GIGL_ROOT_DIR, _make_file_vars["SPARK_35_TFRECORD_JAR_LOCAL_PATH"])
)
SPARK_31_TFRECORD_JAR_LOCAL_PATH: str = str(
    Path.joinpath(GIGL_ROOT_DIR, _make_file_vars["SPARK_31_TFRECORD_JAR_LOCAL_PATH"])
)
