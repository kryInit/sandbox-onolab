from pathlib import Path

project_root_path = Path(__file__).joinpath("../../../").resolve()
datasets_root_path = project_root_path.joinpath("data-storage")
output_path = project_root_path.joinpath("output")
