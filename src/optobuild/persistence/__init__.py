"""Layer 8 - project and result storage.

JSON/YAML project files with schema versioning (ADR-0006). Never uses pickle
for user projects. HDF5 result storage is planned.
"""

from optobuild.persistence.project import (
    Project,
    dumps_project,
    load_project,
    loads_project,
    project_from_dict,
    project_to_dict,
    run_project,
    save_project,
)

__all__ = [
    "Project",
    "dumps_project",
    "load_project",
    "loads_project",
    "project_from_dict",
    "project_to_dict",
    "run_project",
    "save_project",
]
