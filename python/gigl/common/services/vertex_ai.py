import datetime
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    MachineSpec,
    WorkerPoolSpec,
    env_var,
)

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils

logger = Logger()


def _ping_host_ip(host_ip):
    try:
        subprocess.check_output(["ping", "-c", "1", host_ip])
        return True
    except subprocess.CalledProcessError:
        return False


@dataclass
class VertexAiJobConfig:
    job_name: str
    container_uri: str
    command: List[str]
    args: Optional[List[str]] = None
    environment_variables: Optional[List[Dict[str, str]]] = None
    machine_type: str = "n1-standard-4"
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED"
    accelerator_count: int = 0
    replica_count: int = 1
    labels: Optional[Dict[str, str]] = None
    timeout_s: Optional[int] = None
    enable_web_access: bool = True


class VertexAIService:
    """
    A class representing a Vertex AI service.

    Args:
        project (str): The project ID.
        location (str): The location of the service.
        service_account (str): The service account to use for authentication.
        staging_bucket (str): The staging bucket for the service.
    """

    _LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY = "LEADER_WORKER_INTERNAL_IP_FILE_PATH"

    def __init__(
        self, project: str, location: str, service_account: str, staging_bucket: str
    ):
        self.project = project
        self.location = location
        self.service_account = service_account
        self.staging_bucket = staging_bucket
        aiplatform.init(
            project=self.project,
            location=self.location,
            staging_bucket=self.staging_bucket,
        )

    def run(self, job_config: VertexAiJobConfig) -> None:
        """
        Run a Vertex AI job.

        Args:
            job_config (VertexAiJobConfig): The configuration for the job.
        """
        logger.info(f"Running Vertex AI job: {job_config.job_name}")

        machine_spec = MachineSpec(
            machine_type=job_config.machine_type,
            accelerator_type=job_config.accelerator_type,
            accelerator_count=job_config.accelerator_count,
        )

        # This file is used to store the leader worker's internal IP address.
        # Whenever `connect_worker_pool()` is called, the leader worker will
        # write its internal IP address to this file. The other workers will
        # read this file to get the leader worker's internal IP address.
        # See connect_worker_pool() implementation for more details.
        leader_worker_internal_ip_file_path = GcsUri.join(
            self.staging_bucket,
            job_config.job_name,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "leader_worker_internal_ip.txt",
        )
        env_vars = [
            env_var.EnvVar(
                name=VertexAIService._LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY,
                value=leader_worker_internal_ip_file_path.uri,
            )
        ]

        container_spec = ContainerSpec(
            image_uri=job_config.container_uri,
            command=job_config.command,
            args=job_config.args,
            env=env_vars,
        )

        assert (
            job_config.replica_count >= 1
        ), "Replica count can be at minumum 1, i.e. leader worker"

        leader_worker_spec = WorkerPoolSpec(
            machine_spec=machine_spec, container_spec=container_spec, replica_count=1
        )

        worker_pool_specs: List[WorkerPoolSpec] = [leader_worker_spec]

        if job_config.replica_count > 1:
            worker_spec = WorkerPoolSpec(
                machine_spec=machine_spec,
                container_spec=container_spec,
                replica_count=job_config.replica_count - 1,
            )
            worker_pool_specs.append(worker_spec)

        logger.info(
            f"Running Custom job {job_config.job_name} with worker_pool_specs {worker_pool_specs}, in project: {self.project}/{self.location} using staging bucket: {self.staging_bucket}, and attached labels: {job_config.labels}"
        )

        if not job_config.timeout_s:
            logger.info(
                "No timeout set for Vertex AI job, using Vertex AI default timeout of 7 days."
            )
        else:
            logger.info(
                f"Running Vertex AI job with timeout {job_config.timeout_s} seconds"
            )

        job = aiplatform.CustomJob(
            display_name=job_config.job_name,
            worker_pool_specs=worker_pool_specs,
            project=self.project,
            location=self.location,
            labels=job_config.labels,
            staging_bucket=self.staging_bucket,
        )

        job.run(
            service_account=self.service_account,
            timeout=job_config.timeout_s,
            enable_web_access=job_config.enable_web_access,
        )

    @staticmethod
    def is_currently_running_in_vertex_ai_job() -> bool:
        """
        Check if the code is running in a Vertex AI job.

        Returns:
            bool: True if running in a Vertex AI job, False otherwise.
        """
        return VertexAIService.get_vertex_ai_job_id() is not None

    @staticmethod
    def get_vertex_ai_job_id() -> Optional[str]:
        """
        Get the Vertex AI job ID.

        Returns:
            Optional[str]: The Vertex AI job ID, or None if not running in a Vertex AI job.
        """
        return os.getenv("CLOUD_ML_JOB_ID")

    @staticmethod
    def get_host_name() -> Optional[str]:
        """
        Get the current machines hostname.
        """
        return os.getenv("HOSTNAME")

    @staticmethod
    def get_leader_hostname() -> Optional[str]:
        """
        Hostname of the machine that will host the process with rank 0. It is used
        to synchronize the workers.
        """
        return os.getenv("MASTER_ADDR")

    @staticmethod
    def get_leader_port() -> Optional[str]:
        """
        A free port on the machine that will host the process with rank 0.
        """
        return os.getenv("MASTER_PORT")

    @staticmethod
    def get_world_size() -> Optional[str]:
        """
        The total number of processes that VAI creates. Note that VAI only creates one process per machine.
        It is the user's responsibility to create multiple processes per machine.
        """
        return os.getenv("WORLD_SIZE")

    @staticmethod
    def get_rank() -> Optional[str]:
        """
        Rank of the current VAI process, so they will know whether it is the master or a worker.
        Note: that VAI only creates one process per machine. It is the user's responsibility to
        create multiple processes per machine. Meaning, this function will only return one integer
        for the main process that VAI creates.
        """
        return os.getenv("RANK")

    @staticmethod
    def connect_worker_pool() -> str:
        """
        Used to connect the worker pool. This function should be called by all workers
        to get the leader worker's internal IP address and to ensure that the workers
        can all communicate with the leader worker.
        """
        is_leader_worker = VertexAIService.get_rank() == "0"
        ip_file_uri = GcsUri(VertexAIService._get_leader_worker_internal_ip_file_path())
        gcs_utils = GcsUtils()
        host_ip: str
        if is_leader_worker:
            logger.info("Wait 180 seconds for the leader machine to settle down.")
            time.sleep(180)
            host_ip = subprocess.check_output(["hostname", "-i"]).decode().strip()
            logger.info(f"Writing host IP address ({host_ip}) to {ip_file_uri}")
            gcs_utils.upload_from_string(gcs_path=ip_file_uri, content=host_ip)
        else:
            max_retries = 60
            interval_s = 30
            for attempt_num in range(1, max_retries + 1):
                logger.info(
                    f"Checking if {ip_file_uri} exists and reading HOST_IP (attempt {attempt_num})..."
                )
                try:
                    host_ip = gcs_utils.read_from_gcs(ip_file_uri)
                    logger.info(f"Pinging host ip ({host_ip}) ...")
                    if _ping_host_ip(host_ip):
                        logger.info(f"Ping to host ip ({host_ip}) was successful.")
                        break
                except Exception as e:
                    logger.info(e)
                logger.info(
                    f"Retrieving host information and/or ping failed, retrying in {interval_s} seconds..."
                )
                time.sleep(interval_s)
            if attempt_num >= max_retries:
                logger.info(
                    f"Failed to ping HOST_IP after {max_retries} attempts. Exiting."
                )
                raise Exception(f"Failed to ping HOST_IP after {max_retries} attempts.")

        return host_ip

    @staticmethod
    def _get_leader_worker_internal_ip_file_path() -> str:
        """
        Get the file path to the leader worker's internal IP address.
        """
        assert (
            VertexAIService.is_currently_running_in_vertex_ai_job()
        ), "Not running in Vertex AI job."
        internal_ip_file_path = os.getenv(
            VertexAIService._LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY
        )
        assert internal_ip_file_path is not None, (
            f"Internal IP file path ({VertexAIService._LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY}) "
            + f"not found in environment variables. {os.environ}"
        )

        return internal_ip_file_path
