"""Tests for dependency constraints in pyproject.toml.

Ensures that transitive dependencies with known version-sensitivity
are pinned correctly in the project's optional dependency groups.
"""

from importlib.metadata import version
from pathlib import Path
from tomllib import loads as toml_loads

import pytest


def _load_pyproject() -> dict:
    """Load and parse pyproject.toml from the repository root."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    return toml_loads(pyproject_path.read_text())


class TestRagasExtraDependencies:
    """Verify the ragas extra pins critical transitive dependencies."""

    def test_instructor_pinned_in_ragas_extra(self):
        """instructor>=1.0 must appear in the ragas extra.

        Without this pin, uv resolves instructor 0.4.0 which lacks
        instructor.from_anthropic, breaking all LLM-judged metrics.
        See: https://github.com/<org>/raki/issues/134
        """
        pyproject = _load_pyproject()
        ragas_deps = pyproject["project"]["optional-dependencies"]["ragas"]
        instructor_entries = [dep for dep in ragas_deps if dep.lower().startswith("instructor")]
        assert instructor_entries, (
            "instructor is not listed in the ragas extra — "
            "uv will resolve 0.4.0 which breaks LLM-judged metrics"
        )
        # Verify the lower bound is at least 1.0
        dep_spec = instructor_entries[0]
        assert ">=1" in dep_spec or ">1" in dep_spec, (
            f"instructor pin '{dep_spec}' does not enforce >= 1.0"
        )

    def test_installed_instructor_version_is_at_least_1(self):
        """The resolved instructor version must be >= 1.0.

        instructor 0.4.0 does not have instructor.from_anthropic,
        which ragas needs for Anthropic-backed LLM-judged metrics.
        """
        pytest.importorskip("instructor")
        installed_version = version("instructor")
        version_tuple = tuple(int(part) for part in installed_version.split(".")[:2])
        assert version_tuple >= (1, 0), (
            f"instructor {installed_version} is too old — instructor.from_anthropic requires >= 1.0"
        )

    def test_langchain_google_vertexai_removed_from_ragas_extra(self):
        """langchain-google-vertexai must NOT appear in the ragas extra.

        The LLM setup code uses google.genai SDK directly via InstructorLLM and
        GoogleEmbeddings — no LangChain Vertex AI integration is needed.
        langchain-google-vertexai pulls in heavy transitive deps (google-cloud-aiplatform,
        google-cloud-storage, pyarrow, bottleneck) that are unused and slow to install.
        See: https://github.com/<org>/raki/issues/234
        """
        pyproject = _load_pyproject()
        ragas_deps = pyproject["project"]["optional-dependencies"]["ragas"]
        langchain_vertex_entries = [
            dep for dep in ragas_deps if "langchain-google-vertexai" in dep.lower()
        ]
        assert not langchain_vertex_entries, (
            "langchain-google-vertexai is listed in the ragas extra but is no longer used — "
            "the code uses google.genai SDK directly via InstructorLLM / GoogleEmbeddings. "
            f"Remove these entries: {langchain_vertex_entries}"
        )
