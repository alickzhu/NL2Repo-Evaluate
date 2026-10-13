"""AutoNL2Repo generation instruction without controller-only task identifiers."""


def agent_environment_instruction(task, environment_note=""):
    return (
        "According to README.md in /workspace, implement the entire "
        "project as specified. The final product must run directly "
        "from the current workspace, and its API usage must follow "
        "the document's <API Usage Guide>. Work step by step, "
        "test your implementation, and finish only after leaving "
        "all deliverable source files in the workspace.\n\n"
        "You start in a generic environment; choose and install "
        "dependencies, configure the environment, implement, and "
        "test within this same task. "
        "No project dependencies are prefetched; no target "
        "implementation or private tests are provided."
    )

