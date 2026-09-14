# `com.refactoring.extractor` — Eclipse refactoring plugin

This is the **batch refactoring module** of the pipeline (see Section 4.3
of the paper): once the Python ILP CC reducer has selected the Pareto-optimal
sequence of Extract Method operations for a project, this Eclipse plugin
physically applies them to the source code using Eclipse's own JDT/LTK
refactoring engine, and writes out a refactored copy of the project plus a
log of what was done.

## Requirements

- **Eclipse IDE, 2026-06 release (Platform 4.40.0)** — this is not just a
  recommendation: the plugin imports the internal package
  `org.eclipse.jdt.internal.corext.refactoring.code` (marked
  `x-internal` in `MANIFEST.MF`), which is *not* a stable public API and
  can change between Eclipse releases without notice. Running this on a
  different Eclipse version may fail to compile or to behave correctly.
- The **Plug-in Development Environment (PDE)** feature (to import/run/export
  the project) and the **Eclipse Java Development Tools (JDT)** feature.
- **JDK 21** (`Bundle-RequiredExecutionEnvironment: JavaSE-21`).
- The exact bundle versions this plugin was built against (from
  `META-INF/MANIFEST.MF`), useful if you need to check your installed
  Eclipse has compatible versions via *Help → About Eclipse IDE →
  Installation Details → Plug-ins*:

  | Bundle | Version |
  |---|---|
  | `org.eclipse.core.resources` | 3.23.100 |
  | `org.eclipse.core.runtime` | 3.34.100 |
  | `org.eclipse.jdt.core` | 3.44.0 |
  | `org.eclipse.jdt.ui` | 3.36.0 |
  | `org.eclipse.jdt.core.manipulation` | 1.23.200 |
  | `org.eclipse.jdt.launching` | 3.24.0 |
  | `org.eclipse.jface.text` | 3.29.0 |
  | `org.eclipse.ltk.core.refactoring` | 3.15.100 |
  | `org.eclipse.ltk.ui.refactoring` | 3.13.700 |

  Bundled third-party libraries (already included under `lib/`, no separate
  install needed): `commons-csv-1.10.0`, `jackson-core/annotations/databind
  2.15.2`.

## Setup

1. Open Eclipse 2026-06 with PDE + JDT installed.
2. **File → Import… → General → Existing Projects into Workspace**, select
   this `com.refactoring.extractor` folder.
3. Eclipse should resolve all dependencies automatically against your
   installed Eclipse platform (no external Maven/Tycho build is used —
   this is a plain PDE plugin project, compiled/run directly from the
   IDE). If you see unresolved dependency errors, check the version table
   above against *Help → About → Installation Details*.

## Configuration (important — read before running)

Unlike a finished, parameterized tool, the extraction target and options
are currently **hardcoded** at the top of
`src/com/refactoring/extractor/handlers/SampleHandler.java`
(`execute()` method, lines ~29-49). Before each run, edit these fields:

| Field | Meaning |
|---|---|
| `projectSourceDir` | Path to the target project **or** to a "container" folder holding several. Two modes, auto-detected: (1) a single project — a folder **or** a `.zip` of it — processed exactly as before; (2) a folder that is *not* itself a project (no `pom.xml`/`build.xml`/`src` at its top level) is treated as a container of several projects (folders and/or `.zip`s as its direct children), each processed in turn. The original source is never modified either way: a working copy of each project is made in a sibling `<project>_refactored_<algorithm>` folder — created next to the project in single-project mode, or inside the container folder in batch mode. |
| `targetAlgos` | A **list** of algorithms whose solution(s) to apply, e.g. `Arrays.asList("EpsilonConstraintAlgorithm")` (AUGMECON) or `Arrays.asList("EpsilonConstraintAlgorithm", "HybridMethodAlgorithm")` (AUGMECON, then Hybrid Method). With a single entry the behavior is identical to before; with several, every detected project is processed once per algorithm in the list, each (project, algorithm) combination getting its own output folder, log and CSVs. |
| `userPriority` | Objective priority order, e.g. `["loc", "extractions", "cc"]`, used to pick the lexicographic-optimal solution from the Pareto front (see Section 4.3 / "Selection and Injection of Transformations"). |
| `targetClass` | Optional: restrict to a single class (e.g. `"JSON.java"`); leave `""` to process the whole project. |
| `targetSubmodule` | Optional: restrict to a submodule/subfolder of a multi-module project; leave `""` for the whole project. Applied the same way to every project in batch mode, so it's best left `""` unless `projectSourceDir` points at a single project. |
| `resultsBaseDir` | Path to the ILP results produced by the Python reducer (`ilp-cc-reducer/`) — the CSVs listing the candidate extractions this plugin reads and applies. **This is an input location, not where output is written** — see "Running it" below for where the refactored project, log and CSVs actually end up. |

**Batch mode (multiple projects × multiple algorithms):** each
(project, algorithm) combination is processed completely independently —
its own copy of the source, its own log, and its own CSVs (see "Running
it"). If one combination fails (e.g. a broken Maven build for one
project), it's logged and the batch continues with the rest rather than
aborting entirely. Eclipse's internal project registry is also purged
safely between runs, so a previous run's already-refactored output is
never wiped out just because a later run reuses the same project/folder
name — which happens routinely for `.zip`-sourced projects, since the
unzipped internal folder name is the same regardless of which algorithm
is being applied to it.

**Known limitation to flag in the paper/replication package:** this
configuration step is manual (edit source + relaunch), not exposed via a
dialog or CLI flag yet. Paths are also machine-specific (absolute paths).
Anyone reproducing a run needs to edit these fields for their own machine
and target project first.

## Running it

1. **Run → Run As → Eclipse Application.** This launches a second, "runtime"
   Eclipse instance (a temporary workspace — this is the standard way to
   run/debug an Eclipse plugin, and is why you'll see `runtime-*` folders
   next to your workspaces).
2. In that runtime instance, trigger the command either from the
   **Extraction** menu in the main menu bar, its toolbar button, or the
   keyboard shortcut **Ctrl+6** (`M1+6`).
3. The refactoring job runs in the background (so the UI doesn't freeze).
   In batch mode it processes every detected project, once per algorithm
   in `targetAlgos`, one combination at a time. Progress and any errors
   for each combination are printed both to the Eclipse console and to
   that combination's own log file (see below).
4. Each `(project, algorithm)` combination produces its own output folder:
   - **Single-project mode:** `<parent of projectSourceDir>/<project>_refactored_<algorithm>`
   - **Batch mode:** `<container folder>/<project>_refactored_<algorithm>` — one such folder per project, per algorithm, all inside the container you pointed `projectSourceDir` at.

   Inside each of these folders you get:
   - The **refactored copy of the project**, ready to be analyzed with
     SonarQube (see `sonarqube_analysis/` in this replication package) to
     verify the resulting Cognitive Complexity and confirm the
     refactoring compiles and preserves behavior. Extracted methods are
     named `<originalMethod>Extracted<N>` (camelCase) in the source.
   - `<project>[_<class>]_<algorithm>-v2026-06.log` — the full run log.
   - `<project>_<algorithm>_rejected_reasons_v2026-06.csv` — one row per
     extraction that was **not** applied, with the reason (columns:
     `Clase;MetodoOriginal;OffsetInicio;Longitud;MotivoRechazo`).
   - `<project>_<algorithm>_extractions_timing_v2026-06.csv` — one row
     per extraction that was actually **attempted** (successful or not),
     with the generated method's name, the start/end offsets that were
     really applied (after JDT's semantic AST adjustment), how long that
     extraction took in milliseconds, and whether it succeeded (columns:
     `Clase;MetodoOriginal;NombreExtraccion;OffsetInicio;OffsetFin;TiempoEjecucionMs;Exitosa`).
     Useful for a cost/performance analysis of the refactoring process
     itself, independent of the Cognitive Complexity results from
     SonarQube.

## How this fits in the pipeline

`ilp-cc-reducer/` (Python) computes *which* extractions to apply and in
what order; this plugin is what actually *applies* them to real source
code via Eclipse's JDT/LTK refactoring APIs (semantic-preservation checks,
AST manipulation) and produces the refactored projects that were then
re-analyzed with SonarQube — see `sonarqube_analysis/README.md` for that
verification step and the resulting technical-debt figures.