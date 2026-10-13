# Source attribution

This directory is a standalone, reduced local copy of the AutoNL2Repo evaluation
and rollout implementation. The source names below are provenance, not runtime
dependencies or checkout requirements.

- Harness origin: M-Myron/OpenHands_nl2repo, branch
  `repo-generation/curated-rollouts`, commit
  `c016a1595ea48c27640fd6e6ebb343ff37e8c3b7`, subsequently adapted in
  AutoNL2Repo-Eval/repogeneration.
  Its MIT notice is retained at `vendor/LICENSE` (OpenHands contributors).
- OpenHands SDK/tools 1.47.0 are copied under `vendor/`; their original MIT
  license is retained at `vendor/LICENSE`. Package metadata links the upstream
  OpenHands/software-agent-sdk project.
- AutoNL2Repo evaluator 0.3.2 and its test-support material are copied under
  `evaluator/`; exact source hashes are recorded in `PROVENANCE.json`.
- The blocklist is copied from AutoNL2Repo-Eval/openhand/blocklist.py.

- The complete Deckhouse example is included at `data/task.json`. Its frozen
  source/test material originates from `deckhouse/lib-python`, commit
  `a121e758848387c339553309ab08fcf73bd646cb`, under Apache-2.0. The upstream
  license is retained at `data/LICENSE.deckhouse`; this material is not
  relicensed under MIT.

First-party code and documentation are licensed under the root `LICENSE` (MIT).
Third-party license terms and copyright notices remain applicable to their
respective materials. Other task data, scoring images and provider services
have separate licenses and access controls; the root license does not grant
rights to them.
