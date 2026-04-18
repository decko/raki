"""Click CLI entry points for RAKI — run, validate, adapters."""

from pathlib import Path

import click
from rich.console import Console

console = Console()


def _build_registry():
    """Build the default adapter registry with all built-in adapters."""
    from raki.adapters import AdapterRegistry, AlcoveAdapter, SessionSchemaAdapter

    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    registry.register(AlcoveAdapter())
    return registry


def _resolve_manifest(manifest_path: str | None, quiet: bool = False) -> Path:
    """Resolve manifest path from explicit argument or auto-discovery.

    Args:
        manifest_path: Explicit path to manifest file, or None for auto-discovery.
        quiet: When True, suppress the auto-discovery message.

    Returns:
        Resolved path to manifest file.

    Raises:
        click.BadParameter: If the explicit path does not exist.
        click.UsageError: If no manifest is found via auto-discovery.
    """
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
        console.print(f"[dim]Auto-discovered manifest: {found}[/dim]")
    return found


def _warn_unimplemented_options(**options: object) -> None:
    """Print a warning for each option that was provided but is not yet implemented."""
    for option_name, value in options.items():
        if value is not None:
            console.print(f"[yellow]Warning: --{option_name} is not yet implemented[/yellow]")


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
@click.option("--tenant", default=None, help="Set tenant_id on the report")
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
    tenant: str | None,
    include_sessions: bool,
    json_stdout: bool,
    verbose: bool,
) -> None:
    """Run evaluation against sessions."""
    _warn_unimplemented_options(
        adapter=adapter_format,
        metrics=metric_names,
        parallel=parallel_count if parallel_count != 4 else None,
        tenant=tenant,
    )
    manifest_file = _resolve_manifest(manifest_path, quiet=quiet)

    try:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(manifest_file)
    except Exception as exc:
        console.print(f"[red]Error loading manifest: {exc}[/red]")
        raise SystemExit(2) from exc

    from raki.adapters import DatasetLoader

    registry = _build_registry()
    loader = DatasetLoader(registry)

    if not quiet:
        console.print(f"Loading sessions from [bold]{manifest.sessions.path}[/bold]...")

    dataset = loader.load_directory(manifest.sessions.path)

    if verbose:
        for error in loader.errors:
            console.print(f"  [red]Error: {error.path} -- {error.error}[/red]")
        for skipped_path in loader.skipped:
            console.print(f"  [dim]Skipped: {skipped_path}[/dim]")

    if not quiet:
        console.print(
            f"Loaded [bold]{len(dataset.samples)}[/bold] sessions "
            f"({len(loader.skipped)} skipped, {len(loader.errors)} errors)"
        )

    from raki.metrics import MetricsEngine
    from raki.metrics.operational import ALL_OPERATIONAL

    engine = MetricsEngine(ALL_OPERATIONAL)
    report = engine.run(dataset, skip_llm=no_llm)

    if not quiet:
        from raki.report.cli_summary import print_summary

        print_summary(
            report,
            session_count=len(dataset.samples),
            skipped_count=len(loader.skipped),
            error_count=len(loader.errors),
        )

    if threshold is not None and no_llm:
        console.print(
            "[yellow]Warning: --threshold is set but --no-llm is active. "
            "Threshold applies to retrieval quality scores which require LLM metrics.[/yellow]"
        )

    output_path = Path(output_dir)
    from raki.report.json_report import timestamp_filename, write_json_report

    json_file = output_path / timestamp_filename(report)
    write_json_report(report, json_file, include_sessions=include_sessions)

    if json_stdout:
        import json as json_mod

        from raki.report.json_report import _strip_session_data

        data = report.model_dump(mode="json")
        if not include_sessions:
            _strip_session_data(data)
        click.echo(json_mod.dumps(data, indent=2, default=str))

    if not quiet:
        console.print(f"\nReport written:\n  JSON -> {json_file}")

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
@click.option("-v", "--verbose", is_flag=True, help="Show debug output")
def validate(manifest_path: str | None, verbose: bool) -> None:
    """Check manifest and session data without running metrics."""
    manifest_file = _resolve_manifest(manifest_path)

    try:
        from raki.ground_truth.manifest import load_manifest

        manifest = load_manifest(manifest_file)
    except Exception as exc:
        console.print(f"[red]Error loading manifest: {exc}[/red]")
        raise SystemExit(2) from exc
    console.print(f"[green]\u2713[/green] Manifest loaded: {manifest_file}")
    console.print(f"[green]\u2713[/green] Sessions path: {manifest.sessions.path}")

    from raki.adapters import DatasetLoader

    registry = _build_registry()
    loader = DatasetLoader(registry)
    dataset = loader.load_directory(manifest.sessions.path)

    console.print(f"[green]\u2713[/green] {len(dataset.samples)} sessions loaded")

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
            console.print(f"    {error.path.name}: {error.error}")

    console.print(
        f"\nReady to evaluate [bold]{len(dataset.samples)}[/bold] sessions"
        f" ({len(loader.skipped)} skipped, {len(loader.errors)} errors)."
    )


@main.command()
def adapters() -> None:
    """List available session adapters."""
    registry = _build_registry()
    for adapter in registry.list_all():
        console.print(f"  [bold]{adapter.name}[/bold]")


if __name__ == "__main__":
    main()
