import os
import subprocess
from pathlib import Path

from invoke import task

project_root_path = Path(__file__).parent.resolve()
remote_project_root_path = Path('/home/kr/workspace/sandbox-onolab')


@task
def download_output(c):
    subprocess.run(f'scp -r onolab:~/workspace/sandbox-onolab/output {project_root_path}', shell=True)

# @task
# def exec_matlab(c, file_name: str, jp_locale: bool = False):
#     if jp_locale:
#         subprocess.run(f"""ssh onolab 'cd ~/workspace/sspo-exercise/matlab && exec-matlab-jp {file_name}' && poetry run inv download-output > /dev/null""", shell=True)
#     else:
#         subprocess.run(f"""ssh onolab 'cd ~/workspace/sspo-exercise/matlab && exec-matlab {file_name}' && poetry run inv download-output > /dev/null""", shell=True)


@task
def poetry_add(c, name: str, dev: bool = False):
    dev_flag = '-D' if dev else ''
    subprocess.run(f"""poetry add {dev_flag} {name}""", shell=True, cwd=project_root_path)
    sync_venv(c)

@task
def sync_venv(c):
    subprocess.run(f"""scp -r pyproject.toml poetry.lock onolab:{remote_project_root_path} && ssh onolab 'cd {remote_project_root_path} && poetry install'""", shell=True, cwd=project_root_path)


@task
def format(c):
    format_commands = [
        f"isort ./src ./lib ./sandbox",
        f"black ./src ./lib ./sandbox"
    ]

    for cmd in format_commands:
        subprocess.run(cmd, shell=True, cwd=project_root_path)
