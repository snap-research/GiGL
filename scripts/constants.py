from distutils.sysconfig import (  # type: ignore # Since parse_makefile is not discoverable by mypy
    parse_makefile,
)
from pathlib import Path
from typing import Dict

GIGL_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
PATH_GIGL_PKG_INIT_FILE: Path = Path.joinpath(
    GIGL_ROOT_DIR, "python", "gigl", "__init__.py"
)
PATH_BASE_IMAGES_VARIABLE_FILE: Path = Path.joinpath(
    GIGL_ROOT_DIR, "base_images.variable"
).absolute()

_base_image_vars: Dict[str, str] = parse_makefile(PATH_BASE_IMAGES_VARIABLE_FILE)
DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG: str = _base_image_vars[
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG: str = _base_image_vars[
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG: str = _base_image_vars[
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG"
]