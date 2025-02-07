import os
import os.path as osp
import random
import subprocess
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mmengine
from mmengine.config import ConfigDict
from mmengine.utils import track_parallel_progress

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger

from .base import BaseRunner


@RUNNERS.register_module()
class SlurmRunner(BaseRunner):
    """Distributed runner based on Slurm. It will launch tasks in parallel
    using `srun` command.

    Args:
        task (ConfigDict): Task type config.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 32.
        retry (int): Number of retries if the job failed. Defaults to 2.
        partition (str): Slurm partition name. Defaults to None.
        quotatype (str): Slurm quota type. Defaults to None.
        qos (str): Slurm quality of service. Defaults to None.
        debug (bool): Whether to run in debug mode. Defaults to False.
        lark_bot_url (str): Lark bot url. Defaults to None.
        extra_command (List, optional): Extra slurm command.
            For example ['-c 12', '-w node1']. Defaults to None.
    """

    def __init__(self,
                 task: ConfigDict,
                 max_num_workers: int = 32,
                 retry: int = 2,
                 partition: str = None,
                 quotatype: str = None,
                 qos: str = None,
                 debug: bool = False,
                 lark_bot_url: str = None,
                 extra_command: Optional[List[str]] = None):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.retry = retry
        self.partition = partition
        self.quotatype = quotatype
        self.qos = qos
        if not extra_command:
            extra_command = []
        assert isinstance(extra_command, list)
        self.extra_command = extra_command

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """

        if not self.debug:
            status = track_parallel_progress(self._launch,
                                             tasks,
                                             nproc=self.max_num_workers,
                                             keep_order=False)
        else:
            status = [self._launch(task, random_sleep=False) for task in tasks]
        return status

    def _launch(self, cfg: ConfigDict, random_sleep: bool = True):
        """Launch a single task.

        Args:
            cfg (ConfigDict): Task config.
            random_sleep (bool): Whether to sleep for a random time before
                running the command. This avoids cluster error when launching
                multiple tasks at the same time. Default: True.

        Returns:
            tuple[str, int]: Task name and exit code.
        """
        task = TASKS.build(dict(cfg=cfg, type=self.task_cfg['type']))
        num_gpus = task.num_gpus
        task_name = task.name

        # Dump task config to file
        mmengine.mkdir_or_exist('tmp/')
        param_file = f'tmp/{os.getpid()}_params.py'
        try:
            cfg.dump(param_file)

            # Build up slurm command
            tmpl = 'srun'
            if self.partition:
                tmpl += f' -p {self.partition}'
            if self.quotatype:
                tmpl += f' --quotatype={self.quotatype}'
            if self.qos:
                tmpl += f' --qos={self.qos}'
            if num_gpus > 0:
                tmpl += f' --gres=gpu:{num_gpus}'
            for extra_cmd in self.extra_command:
                tmpl += f' {extra_cmd}'
            tmpl += f" -N1 -u -J '{task_name[:512]}'" + ' {task_cmd}'
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            cmd = get_cmd()

            logger = get_logger()
            logger.debug(f'Running command: {cmd}')

            # Run command with retry
            if self.debug:
                stdout = None
            else:
                out_path = task.get_log_path(file_extension='out')
                mmengine.mkdir_or_exist(osp.split(out_path)[0])
                stdout = open(out_path, 'w', encoding='utf-8')

            if random_sleep:
                time.sleep(random.randint(0, 10))
            result = subprocess.run(cmd,
                                    shell=True,
                                    text=True,
                                    stdout=stdout,
                                    stderr=stdout)

            retry = self.retry
            output_paths = task.get_output_paths()
            while self._job_failed(result.returncode,
                                   output_paths) and retry > 0:
                retry -= 1
                if random_sleep:
                    time.sleep(random.randint(0, 10))
                # Re-generate command to refresh ports.
                cmd = get_cmd()
                result = subprocess.run(cmd,
                                        shell=True,
                                        text=True,
                                        stdout=stdout,
                                        stderr=stdout)

            if result.returncode != 0 and not self.debug:
                logger.error(f'task {task_name} fail, see\n{out_path}')
        finally:
            # Clean up
            os.remove(param_file)
        return task_name, result.returncode

    def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
        return return_code != 0 or not all(
            osp.exists(output_path) for output_path in output_paths)
