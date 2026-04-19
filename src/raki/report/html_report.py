"""HTML report generation — self-contained dark-themed report with Jinja2."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jinja2 import Environment, PackageLoader

from raki.model.report import EvalReport
from raki.report.cli_summary import EXPERIMENTAL_METRICS, OPERATIONAL_METRICS


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


def html_color_for_score(score: float, higher_is_better: bool = True) -> str:
    """Return a CSS color class name for a score value.

    Matches the CLI color_for_score semantics: green >= 0.8, yellow >= 0.6, red below.
    """
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


def _build_jinja_env() -> Environment:
    """Create a Jinja2 environment loading templates from the package."""
    return Environment(
        loader=PackageLoader("raki.report", "templates"),
        autoescape=True,
    )


def write_html_report(report: EvalReport, output: Path, include_sessions: bool = False) -> None:
    """Render and write a self-contained HTML report.

    All CSS and JavaScript are inlined in the template — no external dependencies.
    The output file can be opened directly in a browser, shared via email or Slack.

    When include_sessions is False (the default), raw session data is stripped
    from the report before rendering to avoid leaking sensitive information.
    """
    from raki.report.json_report import strip_session_data

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not include_sessions:
        # Strip session data from a serialized copy, then reload as a clean report
        data = report.model_dump(mode="json")
        strip_session_data(data)
        report = EvalReport.model_validate(data)

    operational_scores, retrieval_scores = _split_scores(report.aggregate_scores)
    recurring_failures = _collect_recurring_failures(report)
    worst_sessions = compute_worst_sessions(report, limit=5)

    env = _build_jinja_env()
    template = env.get_template("report.html.j2")

    html_content = template.render(
        report=report,
        operational_scores=operational_scores,
        retrieval_scores=retrieval_scores,
        experimental_metrics=EXPERIMENTAL_METRICS,
        recurring_failures=recurring_failures,
        worst_sessions=worst_sessions,
        color_class=lambda score: f"color-{html_color_for_score(score)}",
        color_name=html_color_for_score,
    )

    output.write_text(html_content)


def html_timestamp_filename(report: EvalReport) -> str:
    """Generate a timestamp-based filename for the HTML report.

    Uses the same datetime format as json_report.timestamp_filename but with .html extension.
    """
    timestamp = report.timestamp
    formatted = timestamp.strftime("%Y%m%dT%H%M%S")
    return f"raki-report-{formatted}.html"
