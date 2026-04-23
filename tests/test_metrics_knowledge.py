"""Tests for knowledge tier metrics: knowledge_gap_rate, knowledge_miss_rate."""

from datetime import datetime, timezone

import pytest

from conftest import make_dataset, make_sample
from raki.docs.chunker import DocChunk
from raki.metrics.protocol import MetricConfig
from raki.model import (
    EvalSample,
    PhaseResult,
    ReviewFinding,
    SessionMeta,
)


# --- STOP_WORDS and word_match ---


class TestWordMatch:
    def test_returns_true_with_three_overlapping_non_stop_words(self):
        """Three domain-specific words in common → True."""
        from raki.metrics.knowledge._common import word_match

        assert (
            word_match(
                "Missing authentication token validation on the endpoint",
                "Authentication token validation must be checked before processing",
            )
            is True
        )

    def test_returns_false_with_fewer_than_three_overlapping_words(self):
        """Only one word in common → False."""
        from raki.metrics.knowledge._common import word_match

        # Only "authentication" overlaps
        assert (
            word_match(
                "Missing authentication check on the endpoint",
                "Authentication tokens must be validated before processing requests",
            )
            is False
        )

    def test_stop_words_are_excluded_from_matching(self):
        """Common stop words do not contribute to overlap count → False."""
        from raki.metrics.knowledge._common import word_match

        assert word_match("the and or if not", "the and or if not be") is False

    def test_matching_is_case_insensitive(self):
        """Upper and lower case tokens are treated identically."""
        from raki.metrics.knowledge._common import word_match

        assert (
            word_match(
                "AUTHENTICATION TOKEN VALIDATION",
                "authentication token validation",
            )
            is True
        )

    def test_stop_words_is_frozenset_containing_common_words(self):
        """STOP_WORDS is a frozenset that filters English function words."""
        from raki.metrics.knowledge._common import STOP_WORDS

        assert isinstance(STOP_WORDS, frozenset)
        assert "the" in STOP_WORDS
        assert "and" in STOP_WORDS
        assert "must" in STOP_WORDS
        assert "be" in STOP_WORDS
        assert "authentication" not in STOP_WORDS
        assert "database" not in STOP_WORDS


# --- KnowledgeGapRate ---


class TestKnowledgeGapRate:
    def test_all_findings_in_uncovered_domains(self):
        """When all findings are in domains not covered by the KB, score is 1.0."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="gap-all",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="information about database schemas and migrations",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication check on the API endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.details["uncovered_findings"] == 1
        assert result.details["total_rework_findings"] == 1

    def test_all_findings_in_covered_domains(self):
        """When all findings are in domains covered by the KB, score is 0.0."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="gap-none",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="Authentication must validate tokens before processing requests",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Missing authentication check on the endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.details["uncovered_findings"] == 0
        assert result.details["total_rework_findings"] == 1

    def test_no_rework_returns_na(self):
        """No rework findings means N/A (score=None)."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        dataset = make_dataset(
            make_sample("1", rework_cycles=0),
            make_sample("2", rework_cycles=0),
        )
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_no_knowledge_context_returns_na(self):
        """When no sessions have knowledge_context, return N/A (score=None)."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="no-kb",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context=None,
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing null check",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.score is None

    def test_mixed_findings(self):
        """Mix of covered and uncovered findings produces correct ratio."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="gap-mixed",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="Authentication tokens must be validated before processing",
        )
        finding_covered = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation",
        )
        finding_uncovered = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Database connection pooling not configured properly",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding_covered, finding_uncovered],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.details["total_rework_findings"] == 2
        assert result.details["uncovered_findings"] == 1
        assert result.score == pytest.approx(0.5)

    def test_ignores_minor_findings(self):
        """Minor findings should not be counted."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="minor-only",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="some knowledge context",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="minor",
            issue="Style nit: use snake_case",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeGapRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_properties(self):
        """Check metric protocol attributes."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        metric = KnowledgeGapRate()
        assert metric.name == "knowledge_gap_rate"
        assert metric.requires_llm is False
        assert metric.requires_ground_truth is False
        assert metric.higher_is_better is False
        assert metric.display_format == "score"
        assert metric.display_name == "Knowledge gap rate"


# --- KnowledgeMissRate ---


class TestKnowledgeMissRate:
    def test_all_findings_in_covered_domains(self):
        """When all findings are in domains covered by KB, score is 1.0 (all misses)."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="miss-all",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="Authentication must validate tokens before processing requests",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Missing authentication check on the endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.score == 1.0
        assert result.details["covered_findings"] == 1
        assert result.details["total_rework_findings"] == 1

    def test_no_findings_in_covered_domains(self):
        """When no findings are in covered domains, score is 0.0."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="miss-none",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="information about database schemas and migrations",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication check on the API endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.score == 0.0
        assert result.details["covered_findings"] == 0
        assert result.details["total_rework_findings"] == 1

    def test_no_rework_returns_na(self):
        """No rework findings means N/A (score=None)."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        dataset = make_dataset(
            make_sample("1", rework_cycles=0),
            make_sample("2", rework_cycles=0),
        )
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_no_knowledge_context_returns_na(self):
        """When no sessions have knowledge_context, return N/A (score=None)."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="no-kb",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context=None,
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing null check",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.score is None

    def test_mixed_findings(self):
        """Mix of covered and uncovered findings produces correct ratio."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="miss-mixed",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="Authentication tokens must be validated before processing",
        )
        finding_covered = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation",
        )
        finding_uncovered = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Database connection pooling not configured properly",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding_covered, finding_uncovered],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.details["total_rework_findings"] == 2
        assert result.details["covered_findings"] == 1
        assert result.score == pytest.approx(0.5)

    def test_ignores_minor_findings(self):
        """Minor findings should not be counted."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="minor-only",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="some knowledge context",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="minor",
            issue="Style nit: use snake_case",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        assert result.score is None
        assert result.details["total_rework_findings"] == 0

    def test_uses_session_phase_fallback(self):
        """Should find knowledge_context from 'session' phase when 'implement' is absent."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="session-phase",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=2,
            rework_cycles=1,
        )
        session_phase = PhaseResult(
            name="session",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="information about authentication and authorization patterns",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication check",
        )
        sample = EvalSample(
            session=meta,
            phases=[session_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)
        result = KnowledgeMissRate().compute(dataset, MetricConfig())
        # "authentication" appears in knowledge_context, so it's a covered finding
        assert result.details["covered_findings"] == 1

    def test_properties(self):
        """Check metric protocol attributes."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        metric = KnowledgeMissRate()
        assert metric.name == "knowledge_miss_rate"
        assert metric.requires_llm is False
        assert metric.requires_ground_truth is False
        assert metric.higher_is_better is False
        assert metric.display_format == "score"
        assert metric.display_name == "Knowledge miss rate"

    def test_doc_chunks_domain_aware_matching(self):
        """With doc chunks, only findings matching a specific domain's content are covered."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="domain-aware",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        # Finding about authentication -- should match auth domain
        finding_auth = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation on the endpoint",
        )
        # Finding about database -- should NOT match auth domain docs
        finding_db = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Database connection pooling not configured properly",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding_auth, finding_db],
            events=[],
        )
        dataset = make_dataset(sample)

        # Only auth domain docs provided -- database finding is NOT covered
        auth_chunks = [
            DocChunk(
                text="Authentication tokens must be validated before processing requests",
                source_file="auth/setup.md",
                domain="auth",
            ),
        ]
        config = MetricConfig(doc_chunks=auth_chunks)
        result = KnowledgeMissRate().compute(dataset, config)

        # Only 1 of 2 findings is in a covered domain (auth)
        assert result.details["covered_findings"] == 1
        assert result.details["total_rework_findings"] == 2
        assert result.score == pytest.approx(0.5)

    def test_doc_chunks_all_domains_covered(self):
        """When doc chunks cover all finding domains, all findings are covered misses."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="all-covered",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)

        auth_chunks = [
            DocChunk(
                text="Authentication tokens must be validated before processing",
                source_file="auth/setup.md",
                domain="auth",
            ),
        ]
        config = MetricConfig(doc_chunks=auth_chunks)
        result = KnowledgeMissRate().compute(dataset, config)
        assert result.score == 1.0
        assert result.details["covered_findings"] == 1

    def test_doc_chunks_no_domains_covered(self):
        """When doc chunks don't cover any finding domain, score is 0.0."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="none-covered",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication check on the endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)

        # Only database docs -- auth finding should NOT be covered
        db_chunks = [
            DocChunk(
                text="Database schemas and migration procedures for PostgreSQL",
                source_file="database/schema.md",
                domain="database",
            ),
        ]
        config = MetricConfig(doc_chunks=db_chunks)
        result = KnowledgeMissRate().compute(dataset, config)
        assert result.score == 0.0
        assert result.details["covered_findings"] == 0
        assert result.details["total_rework_findings"] == 1

    def test_no_doc_chunks_returns_na(self):
        """When no doc chunks are provided, the metric returns N/A (score=None)."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="no-docs",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing null check on the endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)

        # No doc chunks -- metric should return None
        config = MetricConfig(doc_chunks=[])
        result = KnowledgeMissRate().compute(dataset, config)
        assert result.score is None

    def test_doc_chunks_override_knowledge_context(self):
        """When doc chunks are provided, they take precedence over phase knowledge_context."""
        from raki.metrics.knowledge.miss_rate import KnowledgeMissRate

        meta = SessionMeta(
            session_id="override",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        # Phase has knowledge_context covering auth, but doc chunks only cover database
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
            knowledge_context="Authentication tokens must be validated before processing",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)

        # Doc chunks cover database, NOT auth -- finding should NOT be covered
        db_chunks = [
            DocChunk(
                text="Database schemas and migration procedures for PostgreSQL",
                source_file="database/schema.md",
                domain="database",
            ),
        ]
        config = MetricConfig(doc_chunks=db_chunks)
        result = KnowledgeMissRate().compute(dataset, config)
        assert result.score == 0.0
        assert result.details["covered_findings"] == 0


# --- KnowledgeGapRate with doc_chunks ---


class TestKnowledgeGapRateWithDocChunks:
    def test_doc_chunks_domain_aware_matching(self):
        """With doc chunks, only findings NOT matching any domain's content are uncovered."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="gap-domain",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        finding_auth = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing authentication token validation on the endpoint",
        )
        finding_db = ReviewFinding(
            reviewer="ai-review",
            severity="major",
            issue="Database connection pooling not configured properly",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding_auth, finding_db],
            events=[],
        )
        dataset = make_dataset(sample)

        # Only auth domain docs -- database finding is uncovered
        auth_chunks = [
            DocChunk(
                text="Authentication tokens must be validated before processing requests",
                source_file="auth/setup.md",
                domain="auth",
            ),
        ]
        config = MetricConfig(doc_chunks=auth_chunks)
        result = KnowledgeGapRate().compute(dataset, config)

        assert result.details["uncovered_findings"] == 1
        assert result.details["total_rework_findings"] == 2
        assert result.score == pytest.approx(0.5)

    def test_no_doc_chunks_returns_na(self):
        """When no doc chunks are provided, the metric returns N/A (score=None)."""
        from raki.metrics.knowledge.gap_rate import KnowledgeGapRate

        meta = SessionMeta(
            session_id="gap-no-docs",
            started_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            total_phases=3,
            rework_cycles=1,
        )
        implement_phase = PhaseResult(
            name="implement",
            generation=1,
            status="completed",
            output="done",
        )
        finding = ReviewFinding(
            reviewer="ai-review",
            severity="critical",
            issue="Missing null check on the endpoint",
        )
        sample = EvalSample(
            session=meta,
            phases=[implement_phase],
            findings=[finding],
            events=[],
        )
        dataset = make_dataset(sample)

        config = MetricConfig(doc_chunks=[])
        result = KnowledgeGapRate().compute(dataset, config)
        assert result.score is None
