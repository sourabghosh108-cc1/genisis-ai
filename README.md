# Genisis AI

**A multi-domain engineering agent harness combining AI-assisted code
generation, deterministic engineering calculations, and 3D A\* spatial
routing.**

Genisis AI is an experimental Python project developed for the **Google
-- Agent Harness on Physical-Engineering's Last Exam** hackathon. It
explores how an agent workflow can combine language-model assistance
with conventional algorithms and domain-specific calculations for
engineering tasks.

> **Project status:** Prototype / experimental benchmark harness. This
> repository is not a certified engineering design tool and has not been
> validated for real-world construction, safety, or regulatory
> compliance.

## Overview

Genisis AI currently includes:

-   **3D spatial routing:** A grid-based A\* pathfinder that searches
    for a route between two points while avoiding axis-aligned box
    obstacles.
-   **Coordinate transforms:** Drawing-sheet offsets and in-plane
    rotations can be applied to convert local coordinates into a shared
    coordinate space.
-   **Engineering calculations:** Deterministic calculation functions
    for structural beams, chilled-water hydraulics, three-phase
    electrical feeders, and rectangular open-channel flow.
-   **Agent workflow utilities:** A basic plan representation, task
    list, conversation-history compactor, and execution telemetry.
-   **Artifact generation:** CSV predictions, JSON execution
    trajectories, a benchmark chart, and a submission configuration
    file.

## Engineering domains

  -----------------------------------------------------------------------
  Domain                              Current functionality
  ----------------------------------- -----------------------------------
  Mechanical / MEP                    3D pipe routing with obstacle
                                      checks, plus chilled-water
                                      hydraulic calculations

  Structural                          Beam load combinations, shear,
                                      moment, live-load deflection, and
                                      capacity/serviceability comparisons

  Electrical                          Three-phase feeder current and
                                      voltage-drop calculations,
                                      including a configurable 3%
                                      voltage-drop check

  Civil / Hydraulic                   Rectangular-channel area, wetted
                                      perimeter, hydraulic radius,
                                      velocity, discharge, and
                                      Froude-number calculations
  -----------------------------------------------------------------------

The calculations are simplified implementations intended for
experimentation. They do not replace applicable engineering standards,
detailed design software, professional review, or project-specific input
validation.

## How the workflow operates

1.  The harness receives a task identifier, engineering domain, and
    problem description.
2.  It creates a basic plan and task list.
3.  It attempts to request Python code from the configured Google GenAI
    model.
4.  It prepares deterministic reference code for the task.
5.  If the model returns a Python code block that executes successfully,
    that block is selected; otherwise, the deterministic reference code
    is used.
6.  The selected code is executed and its returned metrics are recorded.
7.  The harness creates a report and records estimated token telemetry
    and elapsed time.

**Important:** Successful code execution is not the same as independent
engineering verification. The current report status is an execution
label; it must not be interpreted as proof that every engineering check
passed.

## 3D A\* spatial routing

`SpatialEngineeringEngine` provides a grid-based 3D route search.

Current implementation details:

-   Six axis-aligned movement directions.
-   Configurable grid spacing and search bounds.
-   Axis-aligned box obstacles with a clearance margin.
-   Pipe-radius expansion during collision checks.
-   Sampled segment collision checks.
-   A movement-cost penalty when the route changes direction.
-   Simplification of consecutive collinear path points.

The route search returns a status, waypoints, approximate route length,
number of direction changes, and obstacle count.

**Limitations:** This is a discretized search, not a continuous
geometric optimizer. Collision checking samples points along segments
and can miss cases between samples. Obstacle geometry is represented by
axis-aligned boxes; arbitrary CAD/BIM geometry, pipe fittings,
constructability, and full system coordination are not implemented.

## Engineering calculation library

The `DomainPhysicsLibrary` contains deterministic functions for four
example problem types:

-   **Structural beam analysis:** Calculates factored distributed and
    point loads, maximum moment and shear for a simply supported beam,
    live-load deflection, and comparisons against supplied capacity and
    deflection limits.
-   **Chilled-water hydraulics:** Calculates flow-related quantities and
    head loss using the implemented pipe-flow equations.
-   **Electrical feeder analysis:** Calculates three-phase current and
    voltage drop from the supplied electrical inputs.
-   **Open-channel hydraulics:** Calculates rectangular-channel
    geometry, hydraulic radius, velocity, discharge, and flow regime
    indicators.

All outputs depend on the assumptions and inputs supplied to the
functions. The implementation should be independently checked before any
engineering use.

## Benchmark examples

When no task JSON file is found under `/kaggle/input`, the program uses
four built-in demonstration tasks:

  -----------------------------------------------------------------------
  Task ID                 Domain                  Demonstration problem
  ----------------------- ----------------------- -----------------------
  `PLE_MEP_FLAGSHIP_01`   Mechanical & 3D Spatial Chilled-water pipe
                          Routing                 routing and hydraulic
                                                  calculations

  `PLE_STR_02`            Structural Engineering  Simply supported steel
                                                  girder strength and
                                                  deflection checks

  `PLE_ELEC_03`           Electrical Power        480 V three-phase
                          Systems                 chiller feeder current
                                                  and voltage drop

  `PLE_CIV_04`            Civil & Hydraulic       Rectangular stormwater
                          Engineering             channel calculations
  -----------------------------------------------------------------------

The four built-in examples are a small demonstration set. They should
not be described as a full competition benchmark or as a 29-task
evaluation.

## Observed demonstration results

The development run recorded the following execution telemetry:

  Task ID                   Estimated tokens   Elapsed time
  ----------------------- ------------------ --------------
  `PLE_MEP_FLAGSHIP_01`                  561         1.79 s
  `PLE_STR_02`                           239         0.65 s
  `PLE_ELEC_03`                          198         0.56 s
  `PLE_CIV_04`                           210         0.55 s
  **Total**                        **1,208**     **3.55 s**

-   Average elapsed time: approximately **0.89 seconds per task**.
-   Token counts are estimates based on character length divided by
    four, with small overhead constants. They are not provider-reported
    token usage.
-   In the recorded run, the model label indicated the local
    deterministic engine was used. The results should not be presented
    as proof that the configured cloud model generated the solutions.
-   The structural example produced a moment utilization ratio of
    approximately **1.2077**, which is above 1.0 and therefore indicates
    that the supplied strength-capacity check failed. The harness's
    general `VERIFIED` execution label does not override that
    engineering result.

These figures describe one recorded demonstration run and are not a
reproducibility guarantee or an independently audited benchmark.

## Requirements

The code uses the following Python packages:

-   `numpy`
-   `pandas`
-   `networkx`
-   `matplotlib`
-   `rich`
-   `google-genai` (for the optional Google GenAI request path)

A compatible Python 3 environment is recommended.

## Installation

Clone or download the project, then install the dependencies:

``` bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

``` powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
source .venv/bin/activate
```

Install packages:

``` bash
pip install numpy pandas networkx matplotlib rich google-genai
```

## Configuration

The API key is read from the `GEMINI_API_KEY` environment variable. Do
not hard-code API keys in source files or commit them to version
control.

**Windows PowerShell:**

``` powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
```

**macOS / Linux:**

``` bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

If the Google GenAI client or API request is unavailable, the project
can use its deterministic reference-code path for the built-in examples.

The configured model names are defined near the top of `genisis_ai.py`
and may need to be updated to models available to your API account.

## Run

From the directory containing `genisis_ai.py`:

``` bash
python genisis_ai.py
```

When run successfully, the program writes:

``` text
submission.csv
open_source_harness/
├── submission.yaml
├── benchmark_performance.png
└── agent_trajectories.json
```

The source file also contains code that creates a basic `serve.py`
placeholder and a ZIP bundle. The current `serve.py` is only a startup
scaffold; it is not a complete implementation of the official
competition WebSocket evaluator protocol.

## Output files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `submission.csv`                    Task IDs and generated prediction
                                      reports

  `submission.yaml`                   Example harness submission
                                      configuration

  `benchmark_performance.png`         Execution-time and estimated-token
                                      chart

  `agent_trajectories.json`           Per-task execution details and
                                      telemetry

  `serve.py`                          Minimal server placeholder, not a
                                      complete evaluator server
  -----------------------------------------------------------------------

## Current limitations and safety notes

This project is an experimental prototype. Important limitations
include:

1.  **Execution environment:** `CodeStudioTool` uses Python `exec` with
    unrestricted built-ins. It is not a secure sandbox. Do not execute
    untrusted code with it or expose it to untrusted users.
2.  **Verification claims:** The current workflow reports execution
    status as `VERIFIED` even when a calculated engineering check may
    fail. Treat the metrics---not the generic status string---as the
    relevant calculation output.
3.  **Plan tracking:** The current plan implementation initializes plan
    steps as completed rather than tracking actual step-by-step
    completion.
4.  **Geometry:** Collision checks use axis-aligned boxes and finite
    sampling, not a full CAD/BIM clash-detection engine.
5.  **Model execution:** A configured model name does not prove that the
    model was successfully called or generated the final code. Inspect
    the run log and `model_used` field.
6.  **Benchmark scope:** The fallback task set contains four
    demonstration problems. No 29-task score, cross-model comparison,
    ablation study, or independent reviewer result is claimed here.
7.  **Competition integration:** The generated `serve.py` is a
    placeholder and should not be treated as official evaluator
    integration.

Never use outputs from this prototype as the sole basis for
construction, electrical installation, structural approval, life-safety
decisions, or regulatory submissions. A qualified professional must
independently review all engineering work.

## Project structure

``` text
.
├── genisis_ai.py
├── submission.csv                 # generated after running
└── open_source_harness/            # generated output directory
    ├── submission.yaml
    ├── benchmark_performance.png
    ├── agent_trajectories.json
    └── serve.py
```

## Future development

Potential next steps include:

-   Replace unrestricted `exec` with an isolated execution environment.
-   Add independent calculation checks and explicit PASS / FAIL / REVIEW
    outcomes.
-   Improve geometry handling and continuous collision detection.
-   Track plan steps based on actual execution results.
-   Add automated unit tests and reproducible benchmark fixtures.
-   Implement and test the official competition evaluator interface.
-   Validate calculations against trusted engineering references and
    qualified review.

## Author

**Sourab Ghosh**

Project: **Genisis AI**\
Competition: **Google -- Agent Harness on Physical-Engineering's Last
Exam**

------------------------------------------------------------------------

*Genisis AI is an experimental software project. It is not a substitute
for engineering judgment, professional services, or applicable codes and
standards.*
