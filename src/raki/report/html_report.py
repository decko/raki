"""HTML report generation — self-contained dark-themed report with Jinja2."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from raki.model.report import EvalReport
from raki.report.cli_summary import (
    EXPERIMENTAL_METRICS,
    OPERATIONAL_METRICS,
    generate_summary_sentence,
)

# Metric metadata registry — maps raw metric names to display properties.
# This mirrors the class-level attributes from each Metric implementation
# so the HTML report can render display_name, format values, and pick colors
# without importing the metric classes directly.
METRIC_METADATA: dict[str, dict[str, str | bool]] = {
    "first_pass_verify_rate": {
        "display_name": "Verify rate",
        "higher_is_better": True,
        "display_format": "percent",
        "description": "% sessions passing verify on first try",
        "subtitle": "How often the agent's work passes all checks on the first try",
        "direction": "higher is better",
        "threshold": "Target: >85%",
        "docs_anchor": "verify-rate",
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
    "knowledge_retrieval_miss_rate": {
        "display_name": "Knowledge miss rate",
        "higher_is_better": False,
        "display_format": "score",
        "description": "Fraction of rework caused by missing retrieval context",
        "subtitle": (
            "How often rework happened because the agent lacked the right reference material"
        ),
        "direction": "lower is better",
        "threshold": "Target: <0.20",
        "docs_anchor": "knowledge-miss-rate",
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
    if not higher_is_better and display_format in ("currency", "count"):
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
    aggregate_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Split aggregate scores into operational and retrieval categories."""
    operational = {
        name: score for name, score in aggregate_scores.items() if name in OPERATIONAL_METRICS
    }
    retrieval = {
        name: score for name, score in aggregate_scores.items() if name not in OPERATIONAL_METRICS
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
        for metric_result in sample_result.scores:
            if metric_result.name in OPERATIONAL_METRICS:
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
    When False (--no-llm mode), a footnote is shown instead.
    """
    from raki.report.json_report import strip_session_data

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Compute knowledge context and severity distribution BEFORE stripping
    show_knowledge_miss = has_knowledge_context(report)
    severity_dist = compute_severity_distribution(report)
    cost_range = compute_cost_range(report)

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

    env = _build_jinja_env()
    template = env.get_template("report.html.j2")

    def color_class_fn(score: float, metric_name: str = "") -> str:
        # Special handling for rework_cycles using threshold-based coloring
        if metric_name == "rework_cycles":
            return f"color-{rework_cycles_color(score)}"
        meta = _get_metric_meta(metric_name)
        higher = bool(meta["higher_is_better"])
        fmt = str(meta["display_format"])
        return f"color-{html_color_for_score(score, higher, fmt)}"

    def color_name_fn(score: float, metric_name: str = "") -> str:
        if metric_name == "rework_cycles":
            return rework_cycles_color(score)
        meta = _get_metric_meta(metric_name)
        higher = bool(meta["higher_is_better"])
        fmt = str(meta["display_format"])
        return html_color_for_score(score, higher, fmt)

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
    )

    output.write_text(html_content, encoding="utf-8")


def html_timestamp_filename(report: EvalReport) -> str:
    """Generate a timestamp-based filename for the HTML report.

    Uses the same datetime format as json_report.timestamp_filename but with .html extension.
    """
    timestamp = report.timestamp
    formatted = timestamp.strftime("%Y%m%dT%H%M%S")
    return f"raki-report-{formatted}.html"
