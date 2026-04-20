"""Click CLI entry points for RAKI — run, validate, adapters."""

from __future__ import annotations

import re
from typing import cast
from pathlib import Path

import click
from rich.console import Console

from raki.adapters.redact import redact_sensitive
from raki.report.rerender import is_session_data_stripped, metric_stubs_from_metadata

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

    Includes operational metrics from ALL_OPERATIONAL and Ragas retrieval metrics.
    """
    from raki.metrics.operational import ALL_OPERATIONAL

    names: dict[str, str] = {}
    for metric in ALL_OPERATIONAL:
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
@click.option("--no-llm", is_flag=True, help="Skip LLM-backed metrics")
@click.option("-q", "--quiet", is_flag=True, help="CI mode -- minimal output")
@click.option("--threshold", type=float, default=None, help="Min score for exit code 0")
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
    type=click.Choice(["vertex-anthropic", "anthropic"]),
    default="vertex-anthropic",
    help="LLM provider for judge metrics (default: vertex-anthropic)",
)
@click.option(
    "--include-sessions",
    is_flag=True,
    help="Include full session data in JSON report",
)
@click.option("--json", "json_stdout", is_flag=True, help="Print JSON report to stdout")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output")
def run(
    manifest_path: str | None,
    output_dir: str,
    no_llm: bool,
    quiet: bool,
    threshold: float | None,
    adapter_format: str | None,
    metric_names: str | None,
    parallel_count: int,
    judge_model: str,
    judge_provider: str,
    include_sessions: bool,
    json_stdout: bool,
    verbose: bool,
) -> None:
    """Run evaluation against sessions."""
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

    from raki.metrics import MetricsEngine
    from raki.metrics.operational import ALL_OPERATIONAL
    from raki.metrics.protocol import LLMProvider, Metric, MetricConfig

    config = MetricConfig(
        llm_provider=cast(LLMProvider, judge_provider),
        llm_model=judge_model,
        batch_size=parallel_count,
        project_root=manifest_file.parent.resolve(),
    )

    all_metrics: list[Metric] = list(ALL_OPERATIONAL)

    # Determine whether LLM metrics are needed: only import Ragas machinery
    # when the user has not excluded LLM metrics via --no-llm and the
    # --metrics filter (if any) includes at least one LLM-backed metric.
    needs_llm = not no_llm and (
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
    report = engine.run(dataset, skip_llm=no_llm)

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

    if threshold is not None and no_llm:
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

    if not quiet:
        report_msg = f"\nReport written:\n  JSON -> {json_file}"
        if html_file is not None:
            report_msg += f"\n  HTML -> {html_file}"
        out.print(report_msg)

    if threshold is not None:
        from raki.report.cli_summary import OPERATIONAL_METRICS

        retrieval_scores = {
            name: score
            for name, score in report.aggregate_scores.items()
            if name not in OPERATIONAL_METRICS
        }
        if retrieval_scores:
            mean_retrieval = sum(retrieval_scores.values()) / len(retrieval_scores)
            if mean_retrieval < threshold:
                raise SystemExit(1)


@main.command()
@click.option("-m", "--manifest", "manifest_path", default=None, help="Path to manifest file")
@click.option("-q", "--quiet", is_flag=True, help="Suppress auto-discovery messages")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output")
def validate(manifest_path: str | None, quiet: bool, verbose: bool) -> None:
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
def report(
    input_path: str | None,
    diff_paths: tuple[str, str] | None,
    html_path: str | None,
    output_dir: str | None,
) -> None:
    """Re-render CLI summary and HTML from a saved JSON report.

    Use --diff to compare two evaluation runs side by side.
    """
    if diff_paths is not None:
        _handle_diff(diff_paths, html_path, output_dir)
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


def _handle_diff(
    diff_paths: tuple[str, str],
    html_path: str | None,
    output_dir: str | None,
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


if __name__ == "__main__":
    main()
