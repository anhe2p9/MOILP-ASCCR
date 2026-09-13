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
(`execute()` method, lines ~29-37). Before each run, edit these fields:

| Field | Meaning |
|---|---|
| `projectSourceDir` | Path to the target project — a folder **or** a `.zip` of it. The original is never modified: a working copy is made in a sibling `<project>_refactored_<targetAlgo>` folder. |
| `targetAlgo` | Which algorithm's solution to apply: `"EpsilonConstraintAlgorithm"` (AUGMECON) or `"HybridMethodAlgorithm"` (Hybrid Method). |
| `userPriority` | Objective priority order, e.g. `["loc", "extractions", "cc"]`, used to pick the lexicographic-optimal solution from the Pareto front (see Section 4.3 / "Selection and Injection of Transformations"). |
| `targetClass` | Optional: restrict to a single class (e.g. `"JSON.java"`); leave `""` to process the whole project. |
| `targetSubmodule` | Optional: restrict to a submodule/subfolder of a multi-module project; leave `""` for the whole project. |
| `resultsBaseDir` | Where the refactored output and the run log are written. |

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
3. The refactoring job runs in the background (so the UI doesn't freeze);
   progress and any errors are printed both to the Eclipse console and to
   the log file under `resultsBaseDir` (named
   `<project>[_<class>]_<targetAlgo>-v2026-06.log`).
4. When it finishes, the refactored project is at
   `<parent of projectSourceDir>/<project>_refactored_<targetAlgo>`, ready
   to be analyzed with SonarQube (see `sonarqube_analysis/` in this
   replication package) to verify the resulting Cognitive Complexity and
   confirm the refactoring compiles and preserves behavior.

## How this fits in the pipeline

`ilp-cc-reducer/` (Python) computes *which* extractions to apply and in
what order; this plugin is what actually *applies* them to real source
code via Eclipse's JDT/LTK refactoring APIs (semantic-preservation checks,
AST manipulation) and produces the refactored projects that were then
re-analyzed with SonarQube — see `sonarqube_analysis/README.md` for that
verification step and the resulting technical-debt figures.
