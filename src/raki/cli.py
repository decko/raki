"""Click CLI entry points for RAKI — run, validate, adapters."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
from rich.console import Console

from raki.adapters.redact import redact_sensitive
from raki.report.rerender import is_session_data_stripped, metric_stubs_from_metadata

if TYPE_CHECKING:
    from raki.adapters.loader import DatasetLoader
    from raki.adapters.registry import AdapterRegistry
    from raki.ground_truth.manifest import EvalManifest
    from raki.model import EvalDataset

console = Console()


def _stderr_console() -> Console:
    """Return a Console that writes to stderr, for use when --json is active."""
    return Console(stderr=True)


def _build_registry():
    """Build the default adapter registry with all built-in adapters."""
    from raki.adapters import default_registry

    return default_registry()


def _resolve_manifest(
    manifest_path: str | None, quiet: bool = False, con: Console | None = None
) -> Path:
    """Resolve manifest path from explicit argument or auto-discovery.

    Args:
        manifest_path: Explicit path to manifest file, or None for auto-discovery.
        quiet: When True, suppress the auto-discovery message.
        con: Console to use for output. Falls back to module-level ``console``.

    Returns:
        Resolved path to manifest file.

    Raises:
        click.BadParameter: If the explicit path does not exist.
        click.UsageError: If no manifest is found via auto-discovery.
    """
    out = con or console
    if manifest_path:
        path = Path(manifest_path)
        if not path.exists():
            raise click.BadParameter(f"Manifest not found: {path}")
        return path

    from raki.ground_truth.manifest import discover_manifest

    found = discover_manifest()
    if found is None:
        raise click.UsageError(
            "No manifest found. Provide --manifest or create raki.yaml / eval-manifest.yaml"
        )
    if not quiet:
        out.print(f"[dim]Auto-discovered manifest: {found}[/dim]")
    return found


# LLM metric names that are always known (Ragas-backed retrieval metrics)
_RAGAS_METRICS: dict[str, tuple[str, bool]] = {
    "context_precision": ("Context precision", True),
    "context_recall": ("Context recall", True),
    "faithfulness": ("Faithfulness", True),
    "answer_relevancy": ("Answer relevancy", True),
}


def _all_metric_names() -> dict[str, str]:
    """Return a mapping of all known metric names to display names.

    Includes operational metrics from ALL_OPERATIONAL, knowledge metrics
    from ALL_KNOWLEDGE, and Ragas retrieval metrics.
    """
    from raki.metrics.knowledge import ALL_KNOWLEDGE
    from raki.metrics.operational import ALL_OPERATIONAL

    names: dict[str, str] = {}
    for metric in ALL_OPERATIONAL:
        names[metric.name] = metric.display_name
    for metric in ALL_KNOWLEDGE:
        names[metric.name] = metric.display_name
    for ragas_name, (display_name, _requires_llm) in _RAGAS_METRICS.items():
        names[ragas_name] = display_name
    return names


@click.group()
@click.version_option(package_name="raki")
def main():
    """RAKI -- Retrieval Assessment for Knowledge Impact"""


@main.command()
@click.option("-m", "--manifest", "manifest_path", default=None, help="Path to manifest file")
@click.option("-o", "--output", "output_dir", default="./results", help="Output directory")
@click.option("--judge", is_flag=True, default=False, help="Enable LLM-judged analytical metrics")
@click.option("-q", "--quiet", is_flag=True, help="CI mode -- minimal output")
@click.option(
    "--threshold", type=float, default=None, help="[Deprecated] Min score for exit code 0"
)
@click.option(
    "--gate",
    "gate_thresholds",
    multiple=True,
    help="Quality gate: 'metric>value' (e.g. 'faithfulness>0.85')",
)
@click.option(
    "--require-metric",
    "required_metrics",
    multiple=True,
    help="Fail if metric is N/A instead of skipping threshold",
)
@click.option("--adapter", "adapter_format", default=None, help="Force adapter (default: auto)")
@click.option(
    "--metrics",
    "metric_names",
    default=None,
    help="Comma-separated metric list (default: all)",
)
@click.option(
    "-p",
    "--parallel",
    "parallel_count",
    type=int,
    default=4,
    help="Max parallel LLM calls",
)
@click.option(
    "--judge-model",
    "judge_model",
    default="claude-sonnet-4-6",
    help="LLM model for judge metrics (default: claude-sonnet-4-6)",
)
@click.option(
    "--judge-provider",
    "judge_provider",
    type=click.Choice(["vertex-anthropic", "anthropic", "google", "litellm"]),
    default="vertex-anthropic",
    help="LLM provider for judge metrics (default: vertex-anthropic)",
)
@click.option(
    "--include-sessions",
    is_flag=True,
    help="Include full session data in JSON report",
)
@click.option(
    "--docs-path",
    "docs_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to project docs for knowledge metrics",
)
@click.option("--json", "json_stdout", is_flag=True, help="Print JSON report to stdout")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output")
@click.option(
    "--history-path",
    "history_path_arg",
    default=None,
    help="Path to JSONL history log (default: .raki/history.jsonl)",
)
@click.option(
    "--no-history",
    "no_history",
    is_flag=True,
    default=False,
    help="Skip writing to the JSONL history log",
)
@click.option(
    "--strict-warnings",
    "strict_warnings",
    is_flag=True,
    default=False,
    help="Exit non-zero when metric health errors are detected",
)
def run(
    manifest_path: str | None,
    output_dir: str,
    judge: bool,
    quiet: bool,
    threshold: float | None,
    gate_thresholds: tuple[str, ...],
    required_metrics: tuple[str, ...],
    adapter_format: str | None,
    metric_names: str | None,
    parallel_count: int,
    judge_model: str,
    judge_provider: str,
    docs_path: str | None,
    include_sessions: bool,
    json_stdout: bool,
    verbose: bool,
    history_path_arg: str | None,
    no_history: bool,
    strict_warnings: bool,
) -> None:
    """Run evaluation against sessions."""
    if no_history and history_path_arg is not None:
        raise click.UsageError("--no-history and --history-path cannot be used together.")

    skip_judge = not judge

    out = _stderr_console() if json_stdout else console

    # Validate --metrics filter before doing any heavy loading
    requested_names: set[str] | None = None
    if metric_names is not None:
        all_known = _all_metric_names()
        requested_names = {name.strip() for name in metric_names.split(",")}
        unknown = requested_names - set(all_known.keys())
        if unknown:
            valid_list = ", ".join(sorted(all_known.keys()))
            raise click.BadParameter(
                f"Unknown metric(s): {', '.join(sorted(unknown))}. Valid metrics: {valid_list}",
                param_hint="'--metrics'",
            )

    # Validate --gate metric names before doing any heavy loading
    if gate_thresholds:
        from raki.gates.thresholds import parse_threshold

        all_known = _all_metric_names()
        try:
            parsed_gates_early = [parse_threshold(raw) for raw in gate_thresholds]
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--gate'") from exc
        unknown_gate_metrics = {thr.metric for thr in parsed_gates_early} - set(all_known.keys())
        if unknown_gate_metrics:
            valid_list = ", ".join(sorted(all_known.keys()))
            raise click.BadParameter(
                f"Unknown metric(s) in --gate: {', '.join(sorted(unknown_gate_metrics))}. "
                f"Valid metrics: {valid_list}",
                param_hint="'--gate'",
            )

    manifest_file = _resolve_manifest(manifest_path, quiet=quiet, con=out)

    try:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(manifest_file)
    except Exception as exc:
        out.print(f"[red]Error loading manifest: {redact_sensitive(str(exc))}[/red]")
        raise SystemExit(2) from exc

    from raki.adapters import DatasetLoader

    registry = _build_registry()

    loader = DatasetLoader(registry)

    if not quiet:
        out.print(f"Loading sessions from [bold]{manifest.sessions.path}[/bold]...")

    try:
        dataset = loader.load_directory(manifest.sessions.path, adapter_name=adapter_format)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--adapter'") from exc

    if verbose:
        for error in loader.errors:
            out.print(f"  [red]Error: {error.path} -- {redact_sensitive(error.error)}[/red]")
        for skipped_path in loader.skipped:
            out.print(f"  [dim]Skipped: {skipped_path}[/dim]")

    if not quiet:
        out.print(
            f"Loaded [bold]{len(dataset.samples)}[/bold] sessions "
            f"({len(loader.skipped)} skipped, {len(loader.errors)} errors)"
        )

    # Wire ground truth matching when configured
    if manifest.ground_truth.path is not None:
        try:
            from raki.ground_truth.matcher import load_ground_truth, match_ground_truth

            gt_entries = load_ground_truth(manifest.ground_truth.path)
            matched_count = 0
            for sample in dataset.samples:
                match = match_ground_truth(sample, gt_entries)
                if match is not None:
                    sample.ground_truth = match
                    matched_count += 1
            if not quiet:
                out.print(
                    f"Matched ground truth for "
                    f"[bold]{matched_count}/{len(dataset.samples)}[/bold] sessions"
                )
                if len(dataset.samples) > 0 and matched_count / len(dataset.samples) < 0.5:
                    out.print(
                        "[yellow]Warning: Low match rate — ground truth matching relies on "
                        'output_structured["code_area"] in triage phases. Sessions without '
                        "a triage phase or using a different schema will not match.[/yellow]"
                    )
        except Exception as exc:
            out.print(
                f"[yellow]Warning: Failed to load ground truth: "
                f"{redact_sensitive(str(exc))}. Continuing without ground truth.[/yellow]"
            )

    # Resolve docs path: CLI --docs-path overrides manifest docs.path
    from raki.docs.chunker import DocChunk

    effective_docs_path = Path(docs_path) if docs_path else None
    if effective_docs_path is None and manifest.docs is not None:
        effective_docs_path = manifest.docs.path

    if effective_docs_path is not None:
        resolved_docs = Path(effective_docs_path).resolve()
        project_root = Path.cwd().resolve()
        try:
            resolved_docs.relative_to(project_root)
        except ValueError:
            raise click.UsageError(f"--docs-path must be within the project root ({project_root})")

    doc_chunks: list[DocChunk] = []
    if effective_docs_path is not None:
        from raki.docs.chunker import load_docs

        docs_extensions = None
        if manifest.docs is not None and docs_path is None:
            docs_extensions = manifest.docs.extensions
        doc_chunks = load_docs(effective_docs_path, extensions=docs_extensions)
        if not quiet:
            covered_domains = sorted({chunk.domain for chunk in doc_chunks})
            out.print(
                f"Loaded [bold]{len(doc_chunks)}[/bold] doc chunks "
                f"from {effective_docs_path} "
                f"(domains: {', '.join(covered_domains)})"
            )

    # Wire doc chunks as knowledge_context on each sample's implement/session phase
    if doc_chunks:
        joined_knowledge = "\n---\n".join(chunk.text for chunk in doc_chunks)
        for sample in dataset.samples:
            for phase in sample.phases:
                if phase.name in ("implement", "session") and phase.knowledge_context is None:
                    phase.knowledge_context = joined_knowledge

    from raki.metrics import MetricsEngine
    from raki.metrics.operational import ALL_OPERATIONAL
    from raki.metrics.protocol import LLMProvider, Metric, MetricConfig

    config = MetricConfig(
        llm_provider=cast(LLMProvider, judge_provider),
        llm_model=judge_model,
        batch_size=parallel_count,
        project_root=manifest_file.parent.resolve(),
        doc_chunks=doc_chunks,
    )

    all_metrics: list[Metric] = list(ALL_OPERATIONAL)

    # Add knowledge metrics when docs are loaded
    if doc_chunks:
        from raki.metrics.knowledge import ALL_KNOWLEDGE

        all_metrics.extend(ALL_KNOWLEDGE)

    # Determine whether LLM metrics are needed: only import Ragas machinery
    # when the user has opted in via --judge and the --metrics filter (if any)
    # includes at least one LLM-backed metric.
    needs_llm = not skip_judge and (
        requested_names is None or any(name in requested_names for name in _RAGAS_METRICS)
    )
    if needs_llm:
        from raki.metrics.ragas.faithfulness import FaithfulnessMetric
        from raki.metrics.ragas.precision import ContextPrecisionMetric
        from raki.metrics.ragas.recall import ContextRecallMetric
        from raki.metrics.ragas.relevancy import AnswerRelevancyMetric

        all_metrics.extend(
            [
                ContextPrecisionMetric(),
                ContextRecallMetric(),
                FaithfulnessMetric(),
                AnswerRelevancyMetric(),
            ]
        )

    # Apply --metrics filter after assembling the full metric list
    if requested_names is not None:
        all_metrics = [metric for metric in all_metrics if metric.name in requested_names]

    engine = MetricsEngine(all_metrics, config=config)
    report = engine.run(dataset, skip_judge=skip_judge)

    if not quiet:
        from raki.report.cli_summary import print_summary

        print_summary(
            report,
            session_count=len(dataset.samples),
            skipped_count=len(loader.skipped),
            error_count=len(loader.errors),
            console=out,
            metrics=all_metrics,
        )

    if threshold is not None and skip_judge:
        out.print(
            "[yellow]Warning: No retrieval metrics active — threshold applies only to "
            "LLM-backed metrics. Operational metrics use non-0-1 scales; "
            "per-metric thresholds planned for v0.7.0.[/yellow]"
        )

    output_path = Path(output_dir)
    from raki.report.json_report import timestamp_filename, write_json_report

    json_file = output_path / timestamp_filename(report)
    write_json_report(report, json_file, include_sessions=include_sessions)

    try:
        from raki.report.html_report import html_timestamp_filename, write_html_report

        html_file = output_path / html_timestamp_filename(report)
        write_html_report(
            report,
            html_file,
            include_sessions=include_sessions,
            session_count=len(dataset.samples),
        )
    except ImportError:
        html_file = None
        if not quiet:
            out.print(
                "[yellow]Note: jinja2 not installed — skipping HTML report. "
                "Install with: uv pip install raki[html][/yellow]"
            )

    if json_stdout:
        import json as json_mod

        from raki.report.json_report import strip_session_data

        data = report.model_dump(mode="json")
        if not include_sessions:
            strip_session_data(data)
        click.echo(json_mod.dumps(data, indent=2, default=str))

    # Append to JSONL history log unless the user opted out
    history_path: Path | None = None
    if not no_history:
        history_path = (
            Path(history_path_arg) if history_path_arg else Path.cwd() / ".raki" / "history.jsonl"
        )

        # Path traversal guard: history path must be a descendant of project root
        resolved_history = history_path.resolve()
        project_root = Path.cwd().resolve()
        try:
            resolved_history.relative_to(project_root)
        except ValueError:
            raise click.UsageError(
                f"--history-path must be within the project root ({project_root})"
            )

        from raki.report.history import append_history_entry

        try:
            append_history_entry(
                report,
                history_path,
                sessions_count=len(dataset.samples),
                manifest_file=manifest_file,
            )
        except Exception as exc:
            out.print(f"[yellow]Warning: Failed to write history log: {exc}[/yellow]")
            history_path = None

    if not quiet:
        report_msg = f"\nReport written:\n  JSON -> {json_file}"
        if html_file is not None:
            report_msg += f"\n  HTML -> {html_file}"
        if history_path is not None:
            report_msg += f"\n  History -> {history_path}"
        out.print(report_msg)

    if threshold is not None:
        from raki.report.cli_summary import KNOWLEDGE_METRICS, OPERATIONAL_METRICS

        non_retrieval = OPERATIONAL_METRICS | KNOWLEDGE_METRICS
        retrieval_scores = {
            name: score
            for name, score in report.aggregate_scores.items()
            if name not in non_retrieval and score is not None
        }
        if retrieval_scores:
            mean_retrieval = sum(retrieval_scores.values()) / len(retrieval_scores)
            if mean_retrieval < threshold:
                raise SystemExit(1)

    # Per-metric quality gates (--gate / manifest thresholds)
    effective_thresholds = list(gate_thresholds)
    if not effective_thresholds and manifest.thresholds:
        effective_thresholds = list(manifest.thresholds)

    if effective_thresholds:
        from raki.gates.thresholds import (
            evaluate_all,
            format_threshold_results,
            parse_threshold,
        )

        try:
            parsed_thresholds = [parse_threshold(raw) for raw in effective_thresholds]
        except ValueError as exc:
            out.print(f"[red]Error: {exc}[/red]")
            raise SystemExit(2) from exc

        required_set = set(required_metrics) if required_metrics else None
        gate_results = evaluate_all(
            parsed_thresholds, report.aggregate_scores, required_metrics=required_set
        )

        if not quiet:
            out.print(format_threshold_results(gate_results))

        has_violation = any(not result.passed for result in gate_results)
        if has_violation:
            raise SystemExit(1)

    # --strict-warnings: exit 1 if any metric health errors were detected.
    # Only "error" severity triggers exit code 1; "warning" severity is informational.
    if strict_warnings:
        error_warnings = [w for w in report.warnings if w.severity == "error"]
        if error_warnings:
            if not quiet:
                out.print(
                    f"[red]Strict warnings: {len(error_warnings)} metric health error"
                    f"{'s' if len(error_warnings) > 1 else ''} detected — exiting with code 1[/red]"
                )
            raise SystemExit(1)


@main.command()
@click.option("-m", "--manifest", "manifest_path", default=None, help="Path to manifest file")
@click.option("-q", "--quiet", is_flag=True, help="Suppress auto-discovery messages")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output")
@click.option("--deep", is_flag=True, help="Run smoke-test checks (adapter loading, metrics)")
def validate(manifest_path: str | None, quiet: bool, verbose: bool, deep: bool) -> None:
    """Check manifest and session data without running metrics."""
    manifest_file = _resolve_manifest(manifest_path, quiet=quiet)

    try:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(manifest_file)
    except Exception as exc:
        console.print(f"[red]Error loading manifest: {redact_sensitive(str(exc))}[/red]")
        raise SystemExit(2) from exc
    console.print(f"[green]\u2713[/green] Manifest loaded: {manifest_file}")
    console.print(f"[green]\u2713[/green] Sessions path: {manifest.sessions.path}")

    from raki.adapters import DatasetLoader

    registry = _build_registry()
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(manifest.sessions.path)

    console.print(f"[green]\u2713[/green] {len(dataset.samples)} sessions loaded")

    # Report ground truth status when configured
    if manifest.ground_truth.path is not None:
        try:
            from raki.ground_truth.matcher import load_ground_truth, match_ground_truth

            gt_entries = load_ground_truth(manifest.ground_truth.path)
            console.print(f"[green]\u2713[/green] {len(gt_entries)} ground truth entries loaded")
            preview_matched = sum(
                1
                for sample in dataset.samples
                if match_ground_truth(sample, gt_entries) is not None
            )
            console.print(
                f"[green]\u2713[/green] Ground truth match preview: "
                f"{preview_matched}/{len(dataset.samples)} sessions"
            )
        except Exception as exc:
            console.print(
                f"[yellow]\u26a0[/yellow] Failed to load ground truth: {redact_sensitive(str(exc))}"
            )

    if loader.skipped:
        console.print(
            f"[yellow]\u26a0[/yellow] {len(loader.skipped)} directories skipped (no adapter match)"
        )
        if verbose:
            for skipped_path in loader.skipped:
                console.print(f"    {skipped_path.name}")

    if loader.errors:
        console.print(f"[red]\u2717[/red] {len(loader.errors)} sessions failed to load")
        for error in loader.errors:
            console.print(f"    {error.path.name}: {redact_sensitive(error.error)}")

    console.print(
        f"\nReady to evaluate [bold]{len(dataset.samples)}[/bold] sessions"
        f" ({len(loader.skipped)} skipped, {len(loader.errors)} errors)."
    )

    if deep:
        _run_deep_checks(registry, dataset, manifest, loader)


def _run_deep_checks(
    registry: AdapterRegistry,
    dataset: EvalDataset,
    manifest: EvalManifest,
    loader: DatasetLoader,
) -> None:
    """Run deep smoke-test checks: adapter loading, ground truth, operational metrics.

    No LLM calls, no full evaluation run, no report generation.
    """
    from raki.model import EvalDataset

    console.print("\n[bold]Deep checks[/bold]")

    # --- Check 1: Adapter loading ---
    # Try loading one session through each registered adapter that detected sessions
    if not dataset.samples:
        console.print("[yellow]\u26a0[/yellow] No sessions loaded — skipping adapter checks")
    else:
        session_paths = sorted(p for p in manifest.sessions.path.iterdir() if p.is_dir())
        for adapter in registry.list_all():
            adapter_name = adapter.name
            try:
                # Find a path that this adapter can detect and load
                loaded_any = False
                for session_path in session_paths:
                    if adapter.detect(session_path):
                        sample = adapter.load(session_path)
                        console.print(
                            f"[green]\u2713[/green] Adapter [bold]{adapter_name}[/bold] "
                            f"loaded session: {sample.session.session_id}"
                        )
                        loaded_any = True
                        break
                if not loaded_any:
                    console.print(
                        f"[dim]\u2014[/dim] Adapter [bold]{adapter_name}[/bold] "
                        f"— no matching sessions found"
                    )
            except Exception as exc:
                console.print(
                    f"[red]\u2717[/red] Adapter [bold]{adapter_name}[/bold] "
                    f"failed: {redact_sensitive(str(exc))}"
                )

    # --- Check 2: Ground truth loading and matching ---
    if manifest.ground_truth.path is not None:
        try:
            from raki.ground_truth.matcher import load_ground_truth, match_ground_truth

            gt_entries = load_ground_truth(manifest.ground_truth.path)
            console.print(
                f"[green]\u2713[/green] Ground truth loading: {len(gt_entries)} entries loaded"
            )
            if dataset.samples:
                matched = sum(
                    1
                    for sample in dataset.samples
                    if match_ground_truth(sample, gt_entries) is not None
                )
                console.print(
                    f"[green]\u2713[/green] Ground truth matching: "
                    f"{matched}/{len(dataset.samples)} sessions matched"
                )
            else:
                console.print(
                    "[yellow]\u26a0[/yellow] Ground truth matching: no sessions to match against"
                )
        except Exception as exc:
            console.print(
                f"[red]\u2717[/red] Ground truth loading failed: {redact_sensitive(str(exc))}"
            )

    # --- Check 3: Operational metrics against a single sample ---
    if not dataset.samples:
        console.print(
            "[yellow]\u26a0[/yellow] Operational metrics — no sessions available, skipping"
        )
    else:
        from raki.metrics.operational import ALL_OPERATIONAL
        from raki.metrics.protocol import MetricConfig

        single_dataset = EvalDataset(samples=[dataset.samples[0]])
        config = MetricConfig()
        all_passed = True
        metric_lines: list[str] = []
        for metric in ALL_OPERATIONAL:
            try:
                result = metric.compute(single_dataset, config)
                metric_lines.append(f"    {metric.display_name}: {result.score}")
            except Exception as exc:
                all_passed = False
                metric_lines.append(
                    f"    [red]\u2717[/red] {metric.display_name}: "
                    f"failed — {redact_sensitive(str(exc))}"
                )
        if all_passed:
            console.print(
                "[green]\u2713[/green] Operational metrics — "
                "all computed successfully against 1 sample"
            )
        else:
            console.print("[red]\u2717[/red] Operational metrics — some metrics failed")
        for line in metric_lines:
            console.print(line)


@main.command()
def adapters() -> None:
    """List available session adapters."""
    registry = _build_registry()
    for adapter in registry.list_all():
        console.print(
            f"  [bold]{adapter.name}[/bold]  {adapter.description}  "
            f"[dim](detects: {adapter.detection_hint})[/dim]"
        )


@main.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def metrics(json_output: bool) -> None:
    """List available metrics."""
    from raki.metrics.knowledge import ALL_KNOWLEDGE
    from raki.metrics.operational import ALL_OPERATIONAL

    all_metrics_info: list[dict[str, str | bool]] = []
    for metric in ALL_OPERATIONAL:
        all_metrics_info.append(
            {
                "name": metric.name,
                "display_name": metric.display_name,
                "requires_llm": metric.requires_llm,
                "higher_is_better": metric.higher_is_better,
            }
        )
    for metric in ALL_KNOWLEDGE:
        all_metrics_info.append(
            {
                "name": metric.name,
                "display_name": metric.display_name,
                "requires_llm": metric.requires_llm,
                "higher_is_better": metric.higher_is_better,
            }
        )
    for ragas_name, (display_name, requires_llm) in _RAGAS_METRICS.items():
        all_metrics_info.append(
            {
                "name": ragas_name,
                "display_name": display_name,
                "requires_llm": requires_llm,
                "higher_is_better": True,
            }
        )

    if json_output:
        import json as json_mod

        click.echo(json_mod.dumps({"metrics": all_metrics_info}, indent=2))
        return

    from rich.table import Table

    table = Table(title="Available Metrics")
    table.add_column("Name", style="bold")
    table.add_column("Display Name")
    table.add_column("Requires LLM")
    table.add_column("Higher is Better")
    for info in all_metrics_info:
        table.add_row(
            str(info["name"]),
            str(info["display_name"]),
            "\u2713" if info["requires_llm"] else "",
            "\u2713" if info["higher_is_better"] else "",
        )
    console.print(table)


@main.command()
@click.argument("input_path", required=False, default=None)
@click.option(
    "--diff",
    "diff_paths",
    nargs=2,
    type=str,
    default=None,
    help="Compare two JSON reports: --diff baseline.json compare.json",
)
@click.option(
    "--html",
    "html_path",
    default=None,
    help="Output HTML file path",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    default=None,
    help="Output directory for diff report",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    default=False,
    help="Exit non-zero on metric regression (use with --diff)",
)
@click.option(
    "--gate",
    "gate_thresholds",
    multiple=True,
    help="Quality gate: 'metric>value' (e.g. 'faithfulness>0.85')",
)
@click.option(
    "--require-metric",
    "required_metrics",
    multiple=True,
    help="Fail if metric is N/A instead of skipping threshold",
)
@click.option("-q", "--quiet", is_flag=True, help="CI mode -- minimal output")
def report(
    input_path: str | None,
    diff_paths: tuple[str, str] | None,
    html_path: str | None,
    output_dir: str | None,
    fail_on_regression: bool,
    gate_thresholds: tuple[str, ...],
    required_metrics: tuple[str, ...],
    quiet: bool,
) -> None:
    """Re-render CLI summary and HTML from a saved JSON report.

    Use --diff to compare two evaluation runs side by side.
    Use --gate to apply per-metric quality gates to a saved report.
    """
    if diff_paths is not None:
        _handle_diff(diff_paths, html_path, output_dir, fail_on_regression=fail_on_regression)
        return

    if input_path is None:
        console.print("[red]Error: input path is required when not using --diff[/red]")
        raise SystemExit(2)

    from raki.report.json_report import load_json_report

    path = Path(input_path)
    if not path.exists():
        console.print(f"[red]Error: input file not found: {path}[/red]")
        raise SystemExit(2)

    try:
        eval_report = load_json_report(path)
    except Exception as exc:
        console.print(f"[red]Error loading report: {redact_sensitive(str(exc))}[/red]")
        raise SystemExit(2) from exc

    session_count = len(eval_report.sample_results)
    stripped = is_session_data_stripped(eval_report)

    if not quiet:
        if stripped:
            console.print(
                "[yellow]Warning: Per-session drill-down unavailable — "
                "original report was generated without --include-sessions[/yellow]"
            )

        from raki.report.cli_summary import print_summary

        metric_stubs = metric_stubs_from_metadata(eval_report.aggregate_scores)
        print_summary(
            eval_report,
            session_count=session_count,
            console=console,
            metrics=metric_stubs,
        )

    if html_path is not None:
        try:
            from raki.report.html_report import write_html_report

            write_html_report(
                eval_report,
                Path(html_path),
                include_sessions=not stripped,
                session_count=session_count,
            )
            console.print(f"\nHTML report written: {html_path}")
        except ImportError:
            console.print(
                "[yellow]Note: jinja2 not installed — skipping HTML report. "
                "Install with: uv pip install raki[html][/yellow]"
            )

    if gate_thresholds:
        from raki.gates.thresholds import (
            evaluate_all,
            format_threshold_results,
            parse_threshold,
        )

        try:
            parsed_thresholds = [parse_threshold(raw) for raw in gate_thresholds]
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise SystemExit(2) from exc

        required_set = set(required_metrics) if required_metrics else None
        gate_results = evaluate_all(
            parsed_thresholds, eval_report.aggregate_scores, required_metrics=required_set
        )

        if not quiet:
            console.print(format_threshold_results(gate_results))

        has_violation = any(not result.passed for result in gate_results)
        if has_violation:
            raise SystemExit(1)


def _handle_diff(
    diff_paths: tuple[str, str],
    html_path: str | None,
    output_dir: str | None,
    *,
    fail_on_regression: bool = False,
) -> None:
    """Handle the --diff subflow of the report command."""
    from raki.report.json_report import load_json_report

    baseline_path = Path(diff_paths[0])
    compare_path = Path(diff_paths[1])

    if not baseline_path.exists():
        console.print(f"[red]Error: baseline file not found: {baseline_path}[/red]")
        raise SystemExit(2)
    if not compare_path.exists():
        console.print(f"[red]Error: compare file not found: {compare_path}[/red]")
        raise SystemExit(2)

    try:
        baseline_report = load_json_report(baseline_path)
    except Exception as exc:
        console.print(f"[red]Error loading baseline report: {redact_sensitive(str(exc))}[/red]")
        raise SystemExit(2) from exc

    try:
        compare_report = load_json_report(compare_path)
    except Exception as exc:
        console.print(f"[red]Error loading compare report: {redact_sensitive(str(exc))}[/red]")
        raise SystemExit(2) from exc

    from raki.report.diff import generate_diff_report

    diff_report = generate_diff_report(baseline_report, compare_report)

    from raki.report.cli_summary import print_diff_summary

    print_diff_summary(diff_report, console=console)

    # Determine HTML output path
    resolved_html_path: Path | None = None
    if html_path is not None:
        resolved_html_path = Path(html_path)
    elif output_dir is not None:
        safe_base = re.sub(r"[/\\]|\.\.", "_", baseline_report.run_id)
        safe_comp = re.sub(r"[/\\]|\.\.", "_", compare_report.run_id)
        resolved_html_path = Path(output_dir) / f"diff-{safe_base}-vs-{safe_comp}.html"

    if resolved_html_path is not None:
        try:
            from raki.report.html_report import write_diff_html_report

            write_diff_html_report(diff_report, resolved_html_path)
            console.print(f"\nDiff report written:\n  HTML -> {resolved_html_path}")
        except ImportError:
            console.print(
                "[yellow]Note: jinja2 not installed — skipping HTML diff report. "
                "Install with: uv pip install raki[html][/yellow]"
            )

    # Regression detection gate (--fail-on-regression)
    if fail_on_regression:
        from raki.gates.regression import Direction, compute_exit_code, detect_regressions
        from raki.report.diff import is_higher_is_better

        metric_directions: dict[str, Direction] = {}
        all_metric_names = set(baseline_report.aggregate_scores.keys()) | set(
            compare_report.aggregate_scores.keys()
        )
        for metric_name in all_metric_names:
            if is_higher_is_better(metric_name):
                metric_directions[metric_name] = "higher_is_better"
            else:
                metric_directions[metric_name] = "lower_is_better"

        regression_results = detect_regressions(
            baseline_report.aggregate_scores,
            compare_report.aggregate_scores,
            metric_directions,
        )

        regressed_metrics = [result for result in regression_results if result.regressed]
        if regressed_metrics:
            console.print("\n[red]Regressions detected:[/red]")
            for result in regressed_metrics:
                console.print(
                    f"  {result.metric}: {result.baseline:.4f} -> "
                    f"{result.current:.4f} ({result.direction})"
                )
            exit_code = compute_exit_code(threshold_violated=False, regression_detected=True)
            raise SystemExit(exit_code)


@main.command()
@click.option(
    "--history-path",
    "history_path_arg",
    default=None,
    help="Path to JSONL history log (default: .raki/history.jsonl)",
)
@click.option(
    "--metrics",
    "metric_names",
    default=None,
    help="Comma-separated metric list (default: all)",
)
@click.option(
    "--since",
    "since_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Include only runs on or after this date (YYYY-MM-DD)",
)
@click.option(
    "--until",
    "until_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Include only runs on or before this date (YYYY-MM-DD)",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output as JSON instead of Rich table",
)
@click.option(
    "--last",
    "last_n",
    type=int,
    default=20,
    help="Limit to the last N runs (default: 20; applied after time filters)",
)
@click.option(
    "--manifest",
    "manifest_name",
    type=str,
    default=None,
    help="Filter history entries by manifest name",
)
@click.pass_context
def trends(
    ctx: click.Context,
    history_path_arg: str | None,
    metric_names: str | None,
    since_date: datetime | None,
    until_date: datetime | None,
    json_output: bool,
    last_n: int | None,
    manifest_name: str | None,
) -> None:
    """Show metric trajectories over time from the evaluation history log.

    Reads the JSONL history log and renders a trend table (sparkline + delta)
    for each metric, grouped by tier (Operational → Knowledge → Analytical).

    Use --since / --until to restrict the time window, --metrics to focus on
    specific metrics, and --last to cap the number of runs shown.
    """
    from datetime import timezone

    from raki.report.history import load_history
    from raki.report.trends import compute_all_trends, render_trends_json, render_trends_table

    # Detect whether --last was explicitly passed by the user
    last_explicitly_set = (
        ctx.get_parameter_source("last_n") == click.core.ParameterSource.COMMANDLINE
    )

    # Mutual exclusivity: --since/--until and explicit --last conflict
    if last_explicitly_set and (since_date is not None or until_date is not None):
        raise click.UsageError("--last cannot be combined with --since or --until.")

    # When --since/--until is given, disable the default --last limit
    if not last_explicitly_set and (since_date is not None or until_date is not None):
        last_n = None

    # Validate --metrics names before loading history
    metric_filter: set[str] | None = None
    if metric_names is not None:
        all_known = _all_metric_names()
        requested = {name.strip() for name in metric_names.split(",")}
        unknown = requested - set(all_known.keys())
        if unknown:
            valid_list = ", ".join(sorted(all_known.keys()))
            raise click.BadParameter(
                f"Unknown metric(s): {', '.join(sorted(unknown))}. Valid metrics: {valid_list}",
                param_hint="'--metrics'",
            )
        metric_filter = requested

    # Resolve history path
    history_path = (
        Path(history_path_arg) if history_path_arg else Path.cwd() / ".raki" / "history.jsonl"
    )

    entries = load_history(history_path)

    if not entries:
        console.print("No evaluation history found. Run 'raki run' to generate history.")
        return

    # Apply --last filter (take most recent N entries by timestamp)
    if last_n is not None:
        if last_n < 1:
            raise click.BadParameter("--last must be a positive integer.", param_hint="'--last'")
        entries_sorted = sorted(entries, key=lambda entry: entry.timestamp)
        entries = entries_sorted[-last_n:]

    # Normalize since/until to UTC-aware datetimes for comparison
    since_dt = None
    until_dt = None
    if since_date is not None:
        # click.DateTime returns a naive datetime — treat as UTC start of day
        since_dt = since_date.replace(tzinfo=timezone.utc)
    if until_date is not None:
        until_dt = until_date.replace(tzinfo=timezone.utc)

    trend_list = compute_all_trends(
        entries,
        metric_filter=metric_filter,
        since=since_dt,
        until=until_dt,
        manifest_filter=manifest_name,
    )

    if json_output:
        click.echo(render_trends_json(trend_list))
        return

    render_trends_table(trend_list, console=console)


if __name__ == "__main__":
    main()
