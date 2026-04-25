"""HTML report generation — self-contained dark-themed report with Jinja2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from raki.model.dataset import EvalSample
from raki.model.report import EvalReport, SampleResult
from raki.report.cli_summary import (
    EXPERIMENTAL_METRICS,
    KNOWLEDGE_METRICS,
    OPERATIONAL_METRICS,
    generate_summary_sentence,
)

if TYPE_CHECKING:
    from raki.report.diff import DiffReport

# Metric metadata registry — maps raw metric names to display properties.
# This mirrors the class-level attributes from each Metric implementation
# so the HTML report can render display_name, format values, and pick colors
# without importing the metric classes directly.
METRIC_METADATA: dict[str, dict[str, str | bool]] = {
    "first_pass_success_rate": {
        "display_name": "First-pass success rate",
        "higher_is_better": True,
        "display_format": "percent",
        "description": "% sessions with no rework cycles",
        "subtitle": "How often the agent completes a session without requiring any rework",
        "direction": "higher is better",
        "threshold": "Target: >85%",
        "docs_anchor": "first-pass-success-rate",
    },
    "rework_cycles": {
        "display_name": "Rework cycles",
        "higher_is_better": False,
        "display_format": "count",
        "description": "Average review-fix iterations per session",
        "subtitle": "How many times the agent had to redo its work after feedback",
        "direction": "lower is better",
        "threshold": "Good: <1.0",
        "docs_anchor": "rework-cycles",
    },
    "review_severity_distribution": {
        "display_name": "Severity score",
        "higher_is_better": True,
        "display_format": "score",
        "description": "Weighted severity of review findings (1.0 = no findings)",
        "subtitle": "How many issues reviewers found, broken down by severity",
        "direction": "",
        "threshold": "",
        "docs_anchor": "review-findings",
    },
    "cost_efficiency": {
        "display_name": "Cost / session",
        "higher_is_better": False,
        "display_format": "currency",
        "description": "Average LLM cost per session in USD",
        "subtitle": "How much each agent task costs in API fees",
        "direction": "",
        "threshold": "",
        "docs_anchor": "cost-per-session",
    },
    "self_correction_rate": {
        "display_name": "Self-correction rate",
        "higher_is_better": True,
        "display_format": "percent",
        "description": "Ratio of rework findings resolved by the agent",
        "subtitle": "How often the agent successfully fixes issues found during review",
        "direction": "higher is better",
        "threshold": "Target: >80%",
        "docs_anchor": "self-correction-rate",
    },
    "knowledge_gap_rate": {
        "display_name": "Knowledge gap rate",
        "higher_is_better": False,
        "display_format": "score",
        "description": "Ratio of rework findings in domains not covered by the knowledge base",
        "subtitle": ("How often rework happened because the KB did not cover the relevant domain"),
        "direction": "lower is better",
        "threshold": "Target: <0.20",
        "docs_anchor": "knowledge-gap-rate",
    },
    "knowledge_miss_rate": {
        "display_name": "Knowledge miss rate",
        "higher_is_better": False,
        "display_format": "score",
        "description": "Ratio of rework findings in domains covered by the KB but still wrong",
        "subtitle": (
            "How often the agent got things wrong despite having the right reference material"
        ),
        "direction": "lower is better",
        "threshold": "Target: <0.10",
        "docs_anchor": "knowledge-miss-rate",
    },
    "phase_execution_time": {
        "display_name": "Phase execution time",
        "higher_is_better": False,
        "display_format": "duration",
        "description": "Mean total phase execution time per session (seconds)",
        "subtitle": "How long the agent spends executing phases per session",
        "direction": "lower is better",
        "threshold": "",
        "docs_anchor": "phase-execution-time",
    },
    "token_efficiency": {
        "display_name": "Tokens / phase",
        "higher_is_better": False,
        "display_format": "count",
        "description": "Average tokens (in + out) per phase",
        "subtitle": "How many tokens the agent uses per phase on average",
        "direction": "lower is better",
        "threshold": "",
        "docs_anchor": "token-efficiency",
    },
    "faithfulness": {
        "display_name": "Faithfulness",
        "higher_is_better": True,
        "display_format": "score",
        "description": "Fraction of claims supported by retrieved context",
        "subtitle": ("How closely the agent's output sticks to the facts in its source material"),
        "direction": "higher is better",
        "threshold": "",
        "docs_anchor": "faithfulness",
    },
    "answer_relevancy": {
        "display_name": "Answer relevancy",
        "higher_is_better": True,
        "display_format": "score",
        "description": "How relevant the response is to the user query",
        "subtitle": "How relevant the response is to the user query",
        "direction": "higher is better",
        "threshold": "",
        "docs_anchor": "answer-relevancy",
    },
    "context_precision": {
        "display_name": "Context precision",
        "higher_is_better": True,
        "display_format": "score",
        "description": "Precision of retrieved context relative to ground truth",
        "subtitle": "How much of what the retriever pulled in was actually relevant",
        "direction": "higher is better",
        "threshold": "Target: >0.80",
        "docs_anchor": "context-precision",
    },
    "context_recall": {
        "display_name": "Context recall",
        "higher_is_better": True,
        "display_format": "score",
        "description": "Recall of retrieved context relative to ground truth",
        "subtitle": ("How much of the needed information the retriever successfully found"),
        "direction": "higher is better",
        "threshold": "Target: >0.80",
        "docs_anchor": "context-recall",
    },
}

# GitHub base URL for metric documentation links
DOCS_BASE_URL = "https://github.com/decko/raki/blob/main/docs/interpreting-results.md"


def _get_metric_meta(name: str) -> dict[str, str | bool]:
    """Look up metadata for a metric, falling back to sensible defaults."""
    return METRIC_METADATA.get(
        name,
        {
            "display_name": name,
            "higher_is_better": True,
            "display_format": "score",
            "description": "",
            "subtitle": "",
            "direction": "",
            "threshold": "",
            "docs_anchor": "",
        },
    )


@dataclass(frozen=True)
class DrillDownRow:
    """A row in the per-session drill-down with verdict-based display."""

    session_id: str
    verdict: Literal["pass", "rework", "fail"]
    detail: str
    critical_count: int
    major_count: int
    minor_count: int
    cost: float
    duration_seconds: int
    sort_key: tuple[int, float]


def determine_verdict(sample: EvalSample) -> Literal["pass", "rework", "fail"]:
    """Determine the verdict for a session sample.

    Logic: failed phase -> fail, rework_cycles > 0 -> rework, else pass.
    """
    for phase in sample.phases:
        if phase.status == "failed":
            return "fail"
    if sample.session.rework_cycles > 0:
        return "rework"
    return "pass"


def build_detail(sample: EvalSample) -> str:
    """Build detail text for a drill-down row.

    - Failed phase: "<phase_name> failed"
    - Rework cycles > 0: "<n> cycles"
    - Clean pass: "<n> phases"
    """
    for phase in sample.phases:
        if phase.status == "failed":
            return f"{phase.name} failed"
    if sample.session.rework_cycles > 0:
        return f"{sample.session.rework_cycles} cycles"
    return f"{len(sample.phases)} phases"


def _compute_duration(sample: EvalSample) -> int:
    """Sum phase durations in seconds. Returns 0 if no duration data."""
    total_ms = 0
    for phase in sample.phases:
        if phase.duration_ms is not None:
            total_ms += phase.duration_ms
    return total_ms // 1000


def _format_duration(seconds: int) -> str:
    """Format seconds as M:SS (e.g. 252 -> '4:12')."""
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def compute_drill_down_rows(
    sample_results: list[SampleResult],
) -> list[DrillDownRow]:
    """Build and sort drill-down rows from sample results.

    Sort order: FAIL(0) -> REWORK(1) -> PASS(2), cost descending within each group.
    """
    verdict_rank = {"fail": 0, "rework": 1, "pass": 2}
    rows: list[DrillDownRow] = []

    for sample_result in sample_results:
        sample = sample_result.sample
        session_id = sample.session.session_id
        verdict = determine_verdict(sample)
        detail = build_detail(sample)

        severity_counter: Counter[str] = Counter()
        for finding in sample.findings:
            severity_counter[finding.severity] += 1

        cost = sample.session.total_cost_usd or 0.0
        duration = _compute_duration(sample)

        rows.append(
            DrillDownRow(
                session_id=session_id,
                verdict=verdict,
                detail=detail,
                critical_count=severity_counter.get("critical", 0),
                major_count=severity_counter.get("major", 0),
                minor_count=severity_counter.get("minor", 0),
                cost=cost,
                duration_seconds=duration,
                sort_key=(verdict_rank[verdict], -cost),
            )
        )

    rows.sort(key=lambda row: row.sort_key)
    return rows


@dataclass(frozen=True)
class RecurringFailure:
    """A finding issue that recurs across multiple sessions."""

    issue: str
    severity: Literal["critical", "major", "minor"]
    count: int
    sessions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorstSessionEntry:
    """A session with its average score, used for the worst-sessions shortcut."""

    session_id: str
    avg_score: float


def html_color_for_score(
    score: float,
    higher_is_better: bool = True,
    display_format: str = "score",
) -> str:
    """Return a CSS color class name for a score value.

    Matches the CLI color_for_score semantics: green >= 0.8, yellow >= 0.6, red below.
    Skip color for non-ratio metrics (currency, count) where higher_is_better
    is False -- those values are not on a 0-1 scale.
    """
    if not higher_is_better and display_format in ("currency", "count", "duration"):
        return "white"
    if higher_is_better:
        if score >= 0.8:
            return "green"
        if score >= 0.6:
            return "yellow"
        return "red"
    else:
        if score <= 0.2:
            return "green"
        if score <= 0.4:
            return "yellow"
        return "red"


def _split_scores(
    aggregate_scores: dict[str, float | None],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Split aggregate scores into operational and retrieval categories.

    Knowledge metrics are grouped with operational for display purposes.
    """
    non_retrieval = OPERATIONAL_METRICS | KNOWLEDGE_METRICS
    operational = {name: score for name, score in aggregate_scores.items() if name in non_retrieval}
    retrieval = {
        name: score for name, score in aggregate_scores.items() if name not in non_retrieval
    }
    return operational, retrieval


@dataclass(frozen=True)
class SeverityDistribution:
    """Counts and traffic-light label for review finding severity across all sessions."""

    critical: int
    major: int
    minor: int
    label: Literal["Clean", "Minor", "Moderate", "Severe"]
    total: int

    @property
    def critical_pct(self) -> float:
        """Percentage of critical findings."""
        return (self.critical / self.total * 100) if self.total > 0 else 0.0

    @property
    def major_pct(self) -> float:
        """Percentage of major findings."""
        return (self.major / self.total * 100) if self.total > 0 else 0.0

    @property
    def minor_pct(self) -> float:
        """Percentage of minor findings."""
        return (self.minor / self.total * 100) if self.total > 0 else 0.0


def compute_severity_distribution(report: EvalReport) -> SeverityDistribution:
    """Compute severity distribution from all findings across all sessions.

    Label logic:
    - 0 critical + 0 major = "Clean"
    - 0 critical + some major = "Minor"
    - weighted > 0.5 = "Severe"
    - else = "Moderate"

    Weighted score = (3 * critical + 2 * major + 1 * minor) / (3 * total).
    """
    severity_counter: Counter[str] = Counter()
    for sample_result in report.sample_results:
        for finding in sample_result.sample.findings:
            severity_counter[finding.severity] += 1

    critical_count = severity_counter.get("critical", 0)
    major_count = severity_counter.get("major", 0)
    minor_count = severity_counter.get("minor", 0)
    total_count = critical_count + major_count + minor_count

    if critical_count == 0 and major_count == 0:
        label: Literal["Clean", "Minor", "Moderate", "Severe"] = "Clean"
    elif critical_count == 0:
        label = "Minor"
    else:
        weighted = (3 * critical_count + 2 * major_count + minor_count) / (3 * total_count)
        if weighted > 0.5:
            label = "Severe"
        else:
            label = "Moderate"

    return SeverityDistribution(
        critical=critical_count,
        major=major_count,
        minor=minor_count,
        label=label,
        total=total_count,
    )


def has_knowledge_context(report: EvalReport) -> bool:
    """Check if any session phase has knowledge_context set.

    Returns True if at least one phase in any sample has a non-None knowledge_context.
    """
    for sample_result in report.sample_results:
        for phase in sample_result.sample.phases:
            if phase.knowledge_context is not None:
                return True
    return False


def compute_cost_range(report: EvalReport) -> tuple[float, float] | None:
    """Compute min and max cost from sample session costs.

    Returns None if no sample results have cost data.
    """
    costs: list[float] = []
    for sample_result in report.sample_results:
        cost = sample_result.sample.session.total_cost_usd
        if cost is not None:
            costs.append(cost)
    if not costs:
        return None
    return (min(costs), max(costs))


def rework_cycles_color(value: float) -> str:
    """Return CSS color class for rework cycles based on threshold.

    Green: <1.0, Yellow: 1.0-2.0, Red: >2.0.
    """
    if value < 1.0:
        return "green"
    if value <= 2.0:
        return "yellow"
    return "red"


def collect_agent_models(report: EvalReport) -> list[str]:
    """Collect distinct agent model IDs from sample results, sorted for determinism.

    Returns an empty list when no sample has a model_id set.
    """
    models: set[str] = set()
    for sample_result in report.sample_results:
        model_id = sample_result.sample.session.model_id
        if model_id:
            models.add(model_id)
    return sorted(models)


def _collect_recurring_failures(report: EvalReport) -> list[RecurringFailure]:
    """Find issues that recur across multiple sessions, sorted by count descending."""
    issue_counter: Counter[str] = Counter()
    issue_severity: dict[str, Literal["critical", "major", "minor"]] = {}
    issue_sessions: dict[str, list[str]] = {}

    for sample_result in report.sample_results:
        session_id = sample_result.sample.session.session_id
        for finding in sample_result.sample.findings:
            issue_counter[finding.issue] += 1
            # Keep the highest severity seen for this issue
            existing = issue_severity.get(finding.issue)
            if existing is None or _severity_rank(finding.severity) > _severity_rank(existing):
                issue_severity[finding.issue] = finding.severity
            if finding.issue not in issue_sessions:
                issue_sessions[finding.issue] = []
            if session_id not in issue_sessions[finding.issue]:
                issue_sessions[finding.issue].append(session_id)

    # Only include issues that appear in more than one session
    recurring = []
    for issue_text, count in issue_counter.most_common():
        if len(issue_sessions[issue_text]) > 1:
            recurring.append(
                RecurringFailure(
                    issue=issue_text,
                    severity=issue_severity[issue_text],
                    count=count,
                    sessions=issue_sessions[issue_text],
                )
            )

    return recurring


def _severity_rank(severity: str) -> int:
    """Numeric rank for severity: critical > major > minor."""
    ranks = {"critical": 3, "major": 2, "minor": 1}
    return ranks.get(severity, 0)


def compute_worst_sessions(report: EvalReport, limit: int = 5) -> list[WorstSessionEntry]:
    """Compute the worst-performing sessions by average retrieval metric score.

    Only uses normalized (0-1 range) retrieval metric scores for ranking.
    Operational metrics (which have raw values) are excluded to avoid blending
    scores from different categories. Returns an empty list if no retrieval
    metrics exist.

    Returns at most `limit` sessions sorted by ascending average score.
    """
    entries = []
    for sample_result in report.sample_results:
        session_id = sample_result.sample.session.session_id
        session_scores = []
        non_retrieval = OPERATIONAL_METRICS | KNOWLEDGE_METRICS
        for metric_result in sample_result.scores:
            if metric_result.name in non_retrieval:
                continue
            if session_id in metric_result.sample_scores:
                session_scores.append(metric_result.sample_scores[session_id])
        if session_scores:
            avg = sum(session_scores) / len(session_scores)
            entries.append(WorstSessionEntry(session_id=session_id, avg_score=avg))

    entries.sort(key=lambda entry: entry.avg_score)
    return entries[:limit]


def _build_jinja_env():  # type: ignore[no-any-return]
    """Create a Jinja2 environment loading templates from the package."""
    from jinja2 import Environment, PackageLoader  # ty: ignore[unresolved-import]

    return Environment(
        loader=PackageLoader("raki.report", "templates"),
        autoescape=True,
    )


def write_html_report(
    report: EvalReport,
    output: Path,
    include_sessions: bool = False,
    session_count: int | None = None,
    has_retrieval: bool = True,
) -> None:
    """Render and write a self-contained HTML report.

    All CSS and JavaScript are inlined in the template — no external dependencies.
    The output file can be opened directly in a browser, shared via email or Slack.

    When include_sessions is False (the default), raw session data is stripped
    from the report before rendering to avoid leaking sensitive information.

    session_count is passed as a separate template variable so the header shows
    the correct count even when sample_results is empty.

    has_retrieval controls whether the Retrieval Quality section is shown.
    When False (default mode without --judge), a footnote is shown instead.
    """
    from raki.report.json_report import strip_session_data

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Compute knowledge context, severity distribution, and drill-down BEFORE stripping
    show_knowledge_miss = has_knowledge_context(report)
    severity_dist = compute_severity_distribution(report)
    cost_range = compute_cost_range(report)
    drill_down_rows = compute_drill_down_rows(report.sample_results)
    needs_attention_rows = [row for row in drill_down_rows if row.verdict in ("fail", "rework")]
    needs_attention_count = len(needs_attention_rows)
    agent_models = collect_agent_models(report)

    if not include_sessions:
        # Strip session data from a serialized copy, then reload as a clean report
        data = report.model_dump(mode="json")
        strip_session_data(data)
        report = EvalReport.model_validate(data)

    resolved_session_count = (
        session_count if session_count is not None else len(report.sample_results)
    )

    summary_sentence = generate_summary_sentence(report, resolved_session_count)
    operational_scores, retrieval_scores = _split_scores(report.aggregate_scores)
    recurring_failures = _collect_recurring_failures(report)
    worst_sessions = compute_worst_sessions(report, limit=5)

    no_data_metrics: dict[str, str] = {}
    for metric_name, details in report.metric_details.items():
        if "skipped" in details:
            no_data_metrics[metric_name] = str(details["skipped"])
            continue
        for key, value in details.items():
            if key.startswith("sessions_with_") and value == 0:
                no_data_metrics[metric_name] = "no data in sessions"

    env = _build_jinja_env()
    template = env.get_template("report.html.j2")

    def color_class_fn(score: float | None, metric_name: str = "") -> str:
        if score is None:
            return "color-white"
        # Special handling for rework_cycles using threshold-based coloring
        if metric_name == "rework_cycles":
            return f"color-{rework_cycles_color(score)}"
        meta = _get_metric_meta(metric_name)
        higher = bool(meta["higher_is_better"])
        fmt = str(meta["display_format"])
        return f"color-{html_color_for_score(score, higher, fmt)}"

    def color_name_fn(score: float | None, metric_name: str = "") -> str:
        if score is None:
            return "white"
        if metric_name == "rework_cycles":
            return rework_cycles_color(score)
        meta = _get_metric_meta(metric_name)
        higher = bool(meta["higher_is_better"])
        fmt = str(meta["display_format"])
        return html_color_for_score(score, higher, fmt)

    judge_cost = report.config.get("judge_cost")

    html_content = template.render(
        report=report,
        operational_scores=operational_scores,
        retrieval_scores=retrieval_scores,
        experimental_metrics=EXPERIMENTAL_METRICS,
        recurring_failures=recurring_failures,
        worst_sessions=worst_sessions,
        session_count=resolved_session_count,
        metric_metadata=METRIC_METADATA,
        get_metric_meta=_get_metric_meta,
        color_class=color_class_fn,
        color_name=color_name_fn,
        summary_sentence=summary_sentence,
        severity_dist=severity_dist,
        show_knowledge_miss=show_knowledge_miss,
        cost_range=cost_range,
        has_retrieval=has_retrieval,
        docs_base_url=DOCS_BASE_URL,
        drill_down_rows=drill_down_rows,
        needs_attention_rows=needs_attention_rows,
        needs_attention_count=needs_attention_count,
        format_duration=_format_duration,
        no_data_metrics=no_data_metrics,
        agent_models=agent_models,
        judge_cost=judge_cost,
        metric_warnings=report.warnings,
    )

    output.write_text(html_content, encoding="utf-8")


def write_diff_html_report(
    diff: "DiffReport",
    output: Path,
) -> None:
    """Render and write a self-contained HTML diff report.

    All CSS is inlined — no external dependencies. Uses the same dark theme
    CSS variables as the main report template.
    """
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    match_result = diff.match_result
    matched_count = len(match_result.matched_ids)
    total_count = max(match_result.baseline_total, match_result.compare_total)
    new_count = len(match_result.new_ids)
    dropped_count = len(match_result.dropped_ids)

    def format_display_name(metric_name: str) -> str:
        meta = METRIC_METADATA.get(metric_name, {})
        return str(meta.get("display_name", metric_name))

    def format_value(value: float, metric_name: str) -> str:
        meta = METRIC_METADATA.get(metric_name, {})
        display_format = str(meta.get("display_format", "score"))
        if display_format == "currency":
            return f"${value:.2f}"
        if display_format == "count":
            return f"{value:.1f}"
        if display_format == "percent":
            return f"{value * 100:.0f}%"
        if display_format == "duration":
            return f"{value:.1f}s"
        return f"{value:.2f}"

    def format_delta(delta: float, metric_name: str) -> str:
        meta = METRIC_METADATA.get(metric_name, {})
        display_format = str(meta.get("display_format", "score"))
        sign = "+" if delta >= 0 else ""
        if display_format == "currency":
            if delta < 0:
                return f"-${abs(delta):.2f}"
            return f"{sign}${delta:.2f}"
        if display_format == "count":
            return f"{sign}{delta:.1f}"
        if display_format == "percent":
            return f"{sign}{delta * 100:.0f}%"
        if display_format == "duration":
            return f"{sign}{delta:.1f}s"
        return f"{sign}{delta:.2f}"

    env = _build_jinja_env()
    template = env.get_template("diff.html.j2")

    html_content = template.render(
        diff=diff,
        matched_count=matched_count,
        total_count=total_count,
        new_count=new_count,
        dropped_count=dropped_count,
        new_session_ids=sorted(match_result.new_ids),
        dropped_session_ids=sorted(match_result.dropped_ids),
        format_display_name=format_display_name,
        format_value=format_value,
        format_delta=format_delta,
    )

    output.write_text(html_content, encoding="utf-8")


def html_timestamp_filename(report: EvalReport) -> str:
    """Generate a timestamp-based filename for the HTML report.

    Uses the same datetime format as json_report.timestamp_filename but with .html extension.
    """
    timestamp = report.timestamp
    formatted = timestamp.strftime("%Y%m%dT%H%M%S")
    return f"raki-report-{formatted}.html"
