# BGU HPC / Run:AI --- Operational Knowledge Base for Claude Code

> Purpose: This file is an agent-oriented operational reference for
> working on Ben-Gurion University of the Negev's HPC cluster. Treat the
> official documentation links below as authoritative and live-check
> them before relying on details that may change. This file is a curated
> snapshot, not a complete mirror of every page on the documentation
> site.

## Agent operating contract

When helping with BGU HPC tasks:

1.  **Use official documentation first.** Start at
    <https://hpc.bgu.ac.il/docs/general/index.html>. Follow links to the
    exact workload, setup, FAQ, or CLI page relevant to the task.
    Documentation may change; verify syntax and policies before acting.
2.  **Never run compute workloads, scripts, or autonomous coding agents
    on the login node `bhn20`.** Use it only for the documented
    login/control actions, file transfer, and submitting/managing
    workloads. Do actual code execution and Claude Code work inside an
    approved Run:AI workspace or training workload.
3.  **Prefer BGU-supported browser VS Code for Claude Code.** BGU
    explicitly documents Claude Code installation for VS Code Browser
    workspaces. Do not assume local VS Code Remote SSH and browser VS
    Code use the same setup.
4.  **Do not bypass cluster security or network policy.** Do not edit
    `/etc/hosts`, tunnel around network controls, disable security
    controls, or attempt to work around blocked endpoints. If Anthropic
    connectivity is refused, inspect the official setup and report the
    exact evidence to HPC support.
5.  **Use Conda environments, not the base environment, for
    Python/project dependencies.** BGU's documentation says Conda is
    only available from inside a job/workspace, not on `bhn20`; the base
    environment is not persistent. Keep project environments under the
    recommended persistent location.
6.  **Be resource-conscious.** Request only resources justified by the
    task. Do not allocate a GPU unless needed. Stop/delete unused
    workspaces and trainings according to the documented lifecycle.
7.  **Before destructive actions**, show the exact target and action.
    Never delete a workload, files, environments, or outputs without
    explicit user approval.
8.  **Never invent CLI flags, templates, quotas, paths, or
    availability.** Verify using the linked manual or
    `runai-bgu <command> --help`. If the documentation and installed CLI
    differ, report the discrepancy and prefer cluster-admin guidance.
9.  **Preserve research reproducibility.** Keep code/data/results in
    persistent storage; record environment versions, resource requests,
    command lines, logs, random seeds where relevant, and output paths.
10. **Avoid secrets in prompts/logs.** Do not ask the user to paste
    passwords, access tokens, private keys, or API keys. Use supported
    authentication flows.

## Cluster overview

-   Platform/scheduler: Run:AI, accessed through the BGU-specific
    `runai-bgu` wrapper and the Run:AI web UI.
-   Login/control node documented in examples: `bhn20`.
-   Run:AI projects: users submit workloads under a configured project.
    Use `runai-bgu projects` and `runai-bgu config` to inspect/select
    the permitted project.
-   Two broad workload classes:
    -   **Workspace**: interactive, real-time development, debugging,
        notebooks, data inspection, IDEs.
    -   **Training**: non-interactive batch execution, simulations,
        long-running scripts, and reproducible command-line jobs.
-   The cluster dynamically schedules resources; resource availability
    and queue time are not guaranteed.
-   The BGU documentation notes idle-GPU workspace cleanup after
    approximately six hours without GPU use in the VS Code workspace
    guide. CPU-only jobs/workspaces may continue consuming department
    resources while left running; always check the workload-specific
    page.

## Primary documentation map

-   Main index: <https://hpc.bgu.ac.il/docs/general/index.html>
-   Run:AI workloads overview:
    <https://hpc.bgu.ac.il/docs/general/workloads.html>
-   `runai-bgu` CLI manual:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/manual.html>
-   `runai-bgu submit`:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/submit.html>
-   `runai-bgu list`:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/list.html>
-   `runai-bgu describe`:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/describe.html>
-   `runai-bgu resume`:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/resume.html>
-   `runai-bgu template`:
    <https://hpc.bgu.ac.il/docs/general/runai-bgu/template.html>
-   VS Code workspace (desktop Remote Development):
    <https://hpc.bgu.ac.il/docs/general/workloads/vscode-remote-development.html>
-   Claude Code in VS Code Browser:
    <https://hpc.bgu.ac.il/docs/general/faq/workspace/browser/claude.html>
-   Conda environment management:
    <https://hpc.bgu.ac.il/docs/general/managing-conda-envs.html>
-   Python training setup:
    <https://hpc.bgu.ac.il/docs/general/workload-setup/python-cmd-setup.html>
-   Python training execution:
    <https://hpc.bgu.ac.il/docs/general/workloads/python-cmd.html>
-   Jupyter Notebook workspace:
    <https://hpc.bgu.ac.il/docs/general/workloads/jupyter-notebook.html>
-   PyCharm Remote Development setup:
    <https://hpc.bgu.ac.il/docs/general/workload-setup/pycharm-remote-dev-setup.html>
-   SUMO X11 workspace:
    <https://hpc.bgu.ac.il/docs/general/workloads/sumo-x11.html>
-   Recent documentation changes:
    <https://hpc.bgu.ac.il/docs/general/whats-new.html>

## Supported workload/image names documented for `runai-bgu submit`

The submit manual lists these workload images. Availability and
supported details can vary; follow each workload's own page.

  Workload image   Typical role
  ---------------- -----------------------------------------------------
  `python`         Python batch/training workload
  `pycharm`        PyCharm workspace
  `vscode`         VS Code workspace (desktop Remote Development flow)
  `cmd`            Custom command/batch workload
  `R`              R workload
  `jupyter`        Jupyter workload
  `julia`          Julia workload
  `matlab`         MATLAB workload
  `gmx`            GROMACS
  `namd`           NAMD
  `gurobi`         Gurobi optimization
  `stata`          Stata
  `sumo`           SUMO
  `oneapi`         Intel oneAPI
  `mathematica`    Wolfram Mathematica

The documentation index also mentions browser workspaces or workload
guides for RStudio, Stata, Jupyter, Matlab, ImageJ, KINEROS2, and other
tools. Use the corresponding official workload page rather than assuming
every item is submitted with an identical command.

## `runai-bgu` command map

The CLI manual lists these commands:

-   `bash`: open an interactive shell in a running job.
-   `config`: set the default project.
-   `delete`: delete a workload.
-   `describe`: inspect workload details, resources, pods, events, and
    networking.
-   `list`: list workloads; supports status/project filtering and
    JSON/YAML output in its manual.
-   `login` / `logout`: authenticate/deauthenticate the CLI.
-   `logs`: inspect workload logs/output.
-   `projects`: list available projects.
-   `resume`: resume a suspended workload.
-   `submit`: submit an interactive workspace or batch workload.
-   `template`: open/manage resource templates.
-   Workspace session helpers printed by the CLI include `stop` and
    `stopall`; confirm behavior in the relevant workload guide.

Useful safe discovery commands (run on the documented login/control
path, not as compute tasks):

``` bash
runai-bgu --help
runai-bgu <command> --help
runai-bgu projects
runai-bgu list
runai-bgu describe <workload-name>
runai-bgu logs <workload-name>
runai-bgu template --help
```

Use only commands and options verified by the installed CLI and current
documentation.

### Project selection and authentication

Typical documented flow:

``` bash
runai-bgu login
runai-bgu projects
runai-bgu config <permitted-project-name>
runai-bgu list
```

Do not assume the project name; inspect available projects and confirm
the selected default. Do not store credentials in scripts or repository
files.

### Workload status and troubleshooting

Recommended sequence: 1. `runai-bgu list` to find the workload and
current status. 2. `runai-bgu describe <name>` to inspect status,
resource allocation, pod state, events, and port/network configuration.
3. `runai-bgu logs <name>` to inspect logs. 4. Compare failures against
the exact workload's official page and installed CLI help. 5. If still
unresolved, prepare a concise support report containing the workload
name, timestamp, status/events, command used, and relevant sanitized
logs.

The `list` manual documents options such as `-A` (all projects
accessible to the user), `--status`, `--json`, `--yaml`, `--no-headers`,
`--max-items`, `--page-size`, `--no-pagination`, and `-p <project>`.
Verify current spellings with `runai-bgu list --help`.

## Workload submission and resource options

The `submit` manual documents a general pattern:

``` bash
runai-bgu submit <workload-image> --name <job-name> --cpu <cores> --memory <RAM>
```

Common short forms: - `-n`: name - `-c`: CPU cores - `-m`: RAM,
e.g. `4Gi` - GPU resource can be specified by the GPU option supported
by the relevant command/version. The manual describes `--gpu-memory` as
a VRAM limit; individual workload guides also show `-g` as a
whole/fractional GPU allocation. **Do not treat these as interchangeable
without checking the exact guide and installed CLI.** - RAM and GPU
memory accept units such as `Gi`; `G` and `Gi` are different units. - If
GPU is not needed, omit the GPU request. - Do not set explicit
CPU/RAM/GPU values when using a resource template if the submit manual
says those values conflict with template selection.

The submit manual lists these advanced flags; verify syntax and
availability with the installed CLI: - Diagnostics: `--loglevel`,
`--loglevel-port`, `--loglevel-submit` - Interactive behavior:
`--attach`, `--stdin`, `--tty` - Scheduling/job behavior:
`--preemptible`, `--backoff-limit`, `--completions`, `--parallelism` -
Container/job configuration: `--working-dir`, `--custom-url`,
`--pod-running-timeout`, `--node-pools`, `--node-type`, `--toleration`,
`--environment`, `--project` - Custom command execution: place `--`
followed by the quoted command, with the separating space preserved.
Example:

``` bash
runai-bgu submit python -n example-job -c 2 -m 4Gi --conda myenv -- "python main.py --help"
```

The example is a pattern, not a recommendation to run it without
checking the environment and workload guide.

### Templates

Templates define reusable resource settings: - User template:
`--ut <template-name>` - Group template: documentation examples use
`--gt <template-name>` in the `submit` manual; some workload pages show
`--ug`. **This is a documentation inconsistency. Verify the current CLI
help and template guide before choosing the flag; do not silently
guess.** - Open the template interface with `runai-bgu template`; use
`runai-bgu template --help` and the official template guide to
list/create/edit/delete templates. - When submitting with a template, do
not also specify resource flags that the current manual says conflict
with templates.

## Workspace lifecycle and cost/resource hygiene

For a VS Code workspace, the official guide documents: -
`runai-bgu submit vscode -n <name> -c <cores> -m <RAM>` for explicit
CPU/RAM. - Add GPU allocation only when required and only using the
supported syntax. - The CLI prints a port-forwarding port for desktop
Remote Development; use the assigned port, never guess one. - `stopall`:
stops forwarding and deletes the workspace (per VS Code guide). -
`stop`: stops port forwarding but leaves the workload running. This
continues consuming allocated resources. - If a workload remains
running, the guide says to repeat the submission/connection steps to
re-establish forwarding. - If suspended, use `runai-bgu resume <name>`;
then follow the reconnection steps. - A suspended workload can queue
until resources are available. Check `list`/`describe` afterward. -
Verify persistence of files in the documented persistent storage. Do not
assume ephemeral container paths survive deletion.

Before using `stop`, `stopall`, `delete`, or `resume`, tell the user
what the command does and which named workload it targets. `stopall` is
destructive to the workspace workload itself, even though files in
persistent home storage may remain.

## VS Code: two different workflows

### A. Desktop VS Code Remote Development

Official guide:
<https://hpc.bgu.ac.il/docs/general/workloads/vscode-remote-development.html>

-   SSH to the login/control node as documented, then submit a `vscode`
    workspace.
-   The CLI starts the workspace and reports a port used for forwarding.
-   Configure VS Code Remote Development/SSH using the actual host,
    user, key, and port according to the official guide.
-   Do not run the coding agent on `bhn20`; the agent and project
    processes must run inside the approved workspace.
-   This is distinct from opening a browser-based VS Code workspace.

### B. Browser VS Code workspace + Claude Code (BGU's recommended documented route)

Official guide:
<https://hpc.bgu.ac.il/docs/general/faq/workspace/browser/claude.html>

The BGU guide states:

``` bash
npm install -g @anthropic-ai/claude-code@2.1.89
/gpfs0/runai-utils/setup-claude-vscode-open-folder.sh
```

It warns that some Claude Code versions are broken and identifies
`2.1.89` as the latest tested version on that page. Do not automatically
upgrade to a newer version without checking current BGU instructions.

If `npm` is unavailable in the Conda environment, the guide recommends
creating an environment with Node.js 24 (Claude Code requires Node.js
\>=22):

``` bash
conda create -n claude-env nodejs=24 -c conda-forge
conda activate claude-env
npm install -g @anthropic-ai/claude-code@2.1.89
/gpfs0/runai-utils/setup-claude-vscode-open-folder.sh
```

Run these inside the approved browser VS Code workspace terminal, not on
`bhn20`. The BGU script is part of the documented setup; do not replace
it with guessed settings.

Claude Code still requires supported authentication and account/billing
eligibility. Installing the CLI does not itself grant access. If
authentication or network connection fails: - Capture the exact error
and relevant sanitized diagnostics. - Check the current BGU Claude FAQ
and common-errors guide. - Do not edit `/etc/hosts`, override DNS, route
around proxy/firewall rules, or try to bypass network restrictions. -
Escalate to BGU HPC support if the official documented flow cannot reach
the required service.

**Known user-specific troubleshooting observation (verify before
acting):** in an earlier session, `api.anthropic.com` resolved to
`0.0.0.0` and `/etc/hosts` on both `bhn20` and the workspace showed
entries mapping Anthropic domains to `0.0.0.0`. This suggests a
policy/configuration issue rather than a normal Claude account problem.
Do not attempt to remove or bypass those mappings. Report this evidence
to HPC support and ask for the approved supported route.

## Conda, Python and reproducibility

Official guide:
<https://hpc.bgu.ac.il/docs/general/managing-conda-envs.html>

Important documented points: - `conda` is not available on login node
`bhn20`; use a job/workspace terminal. - Configure the default
environment directory under `~/env` as documented:

``` bash
conda config --prepend envs_dirs ~/env
conda config --show envs_dirs
```

-   Example environment operations:

``` bash
conda create --name myenv python=3.11
conda activate myenv
conda deactivate
conda env list
```

-   Use named environments and avoid relying on the base environment.
-   For Jupyter, a Conda environment does not necessarily appear
    automatically as a notebook kernel; follow the official
    kernel-registration guide linked from the Jupyter page.
-   For Python training workloads, make scripts CLI-runnable and submit
    a command that activates/uses the intended environment and executes
    the script. Follow the Python setup page and the Python training
    page.

Example Python training pattern (adapt only after confirming the project
environment and resource needs):

``` bash
runai-bgu submit python -n <job-name> -c <cores> -m <RAM> --conda <conda-env> -- "python <script.py> <args>"
```

The official Python guide uses `&&` to join multiple shell steps in the
command string. Preserve the `-- "command"` structure.

## Other documented workflow families

Use the dedicated official pages before creating workloads for these: -
Browser workspaces: Jupyter Notebook, PyCharm, RStudio, Matlab, Stata. -
X11/desktop-style workspaces: SUMO, Stata. - Batch/trainings: Python,
Mathematica, Matlab, R, Julia, GROMACS, NAMD, CMD/custom commands,
Gurobi, oneAPI, Jupyter, ImageJ, KINEROS2. - Additional software listed
on the main index: GROMACS, NAMD, Stata, R, RStudio, Python, Jupyter,
Julia, MATLAB, VS Code, PyCharm, SUMO, Mathematica, Gurobi, oneAPI,
ImageJ, and a universal code image for languages supported by Conda.

This list is not a guarantee of current availability or entitlement.
Follow the page for the specific tool and the current resource policy.

## Security, policy, and support

-   Never run CPU/GPU-intensive work or coding agents directly on
    `bhn20`.
-   Do not use `sudo` or attempt privileged system modifications unless
    HPC administrators explicitly direct it.
-   Do not modify cluster-managed files or global environment
    configuration.
-   Do not expose secrets in command history, logs, notebooks,
    repositories, or prompts.
-   Do not assume a workload has GPU access just because a GPU-related
    package is installed.
-   Do not assume a GPU quota or resource pool; check current
    documentation, project policy, `describe`, and administrator
    guidance.
-   When a requested task conflicts with cluster rules, explain the
    constraint and offer a compliant workflow.
-   If the docs conflict or omit an operational detail, inspect current
    `--help` and contact HPC support instead of guessing.

## Agent execution protocol

For any non-trivial HPC task, Claude should:

1.  Classify it as (a) code editing, (b) interactive debugging, (c)
    batch compute, (d) data transfer, or (e) workload management.
2.  Identify which actions must happen on the login/control node and
    which must happen inside a workspace/training.
3.  Read the exact official BGU page and verify installed CLI help.
4.  Inspect project directory and current workload status before making
    changes.
5.  Propose the resource request and explain CPU/RAM/GPU assumptions.
6.  Ask before destructive operations, expensive resource allocations,
    environment changes that could affect other work, or deleting
    outputs.
7.  Execute only inside the approved workload.
8.  Capture command, exit status, logs, environment/version information,
    resource configuration, and output paths.
9.  Verify results and report what actually succeeded versus what
    remains unverified.
10. If blocked by policy/network/permissions, stop attempting
    workarounds and produce a support-ready diagnostic report.

## Citation / freshness rule

When answering a BGU HPC question, link the exact official documentation
page used. Distinguish: - **Documented fact**: directly supported by an
official BGU page. - **Observed local fact**: seen in the current
terminal/workload, with timestamp/context. - **Inference**: plausible
explanation, clearly labeled. - **Unknown**: must be checked with the
CLI or HPC support.

Last curated: 2026-10-13. This document should be revalidated against
the live official site before relying on version-specific or
policy-sensitive details.
