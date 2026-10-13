"""Verbatim AutoNL2Repo development preparation; no historical scorer."""
from ..dependencies import DependencyWheelhouse
from ..grading import prepare_task


class AutoNL2RepoAdapter:
    def prepare_work(self, target, task, *, work_image, blocked_packages):
        prepare_task(
            target,
            task,
            work_image=work_image,
            blocked_packages=blocked_packages,
            empty_workspace=True,
        )
        DependencyWheelhouse({}, {}).stage(target)
        target.write_file("/workspace/README.md", task.public.prompt)
        return (
            "Implement the repository described below in /workspace. You start in a "
            "generic environment and must choose and install dependencies yourself. "
            "The Specification is also available as README.md. No tests, reference "
            "implementation or private build configuration are provided.\n\n"
            + task.public.prompt
        )

