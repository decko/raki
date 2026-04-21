"""Tests for the docs chunker module.

Covers markdown chunking, RST chunking, plain text chunking,
path safety, and domain extraction.
"""

from pathlib import Path

from raki.docs.chunker import chunk_file, load_docs


class TestMarkdownChunking:
    """Tests for Markdown heading-based chunking."""

    def test_split_on_h1_headings(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        md_file.write_text("# Introduction\nSome intro text.\n# Setup\nSetup instructions.\n")
        chunks = chunk_file(md_file, tmp_path)
        assert len(chunks) == 2
        assert chunks[0].text.startswith("# Introduction")
        assert "Some intro text." in chunks[0].text
        assert chunks[1].text.startswith("# Setup")
        assert "Setup instructions." in chunks[1].text

    def test_split_on_h2_headings(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        md_file.write_text("## First Section\nContent one.\n## Second Section\nContent two.\n")
        chunks = chunk_file(md_file, tmp_path)
        assert len(chunks) == 2
        assert chunks[0].text.startswith("## First Section")
        assert chunks[1].text.startswith("## Second Section")

    def test_heading_included_in_chunk(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        md_file.write_text("# My Heading\nParagraph content here.\n")
        chunks = chunk_file(md_file, tmp_path)
        assert len(chunks) == 1
        assert "# My Heading" in chunks[0].text
        assert "Paragraph content here." in chunks[0].text

    def test_no_heading_fallback_single_chunk(self, tmp_path: Path) -> None:
        md_file = tmp_path / "notes.md"
        md_file.write_text("Just some text without any headings.\nAnother line.\n")
        chunks = chunk_file(md_file, tmp_path)
        assert len(chunks) == 1
        assert "Just some text" in chunks[0].text

    def test_mixed_heading_levels(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        content = "# Top\nIntro.\n## Sub\nDetails.\n### Deep\nMore.\n"
        md_file.write_text(content)
        chunks = chunk_file(md_file, tmp_path)
        assert len(chunks) == 3
        assert chunks[0].text.startswith("# Top")
        assert chunks[1].text.startswith("## Sub")
        assert chunks[2].text.startswith("### Deep")

    def test_source_file_set_correctly(self, tmp_path: Path) -> None:
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\nContent.\n")
        chunks = chunk_file(md_file, tmp_path)
        assert chunks[0].source_file == "readme.md"

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        md_file = tmp_path / "empty.md"
        md_file.write_text("")
        chunks = chunk_file(md_file, tmp_path)
        assert chunks == []

    def test_whitespace_only_file_returns_empty(self, tmp_path: Path) -> None:
        md_file = tmp_path / "blank.md"
        md_file.write_text("   \n  \n  ")
        chunks = chunk_file(md_file, tmp_path)
        assert chunks == []


class TestRSTChunking:
    """Tests for RST underline heading detection."""

    def test_rst_equals_underline(self, tmp_path: Path) -> None:
        rst_file = tmp_path / "guide.rst"
        rst_file.write_text(
            "Introduction\n============\nIntro text.\n\nSetup\n=====\nSetup text.\n"
        )
        chunks = chunk_file(rst_file, tmp_path)
        assert len(chunks) == 2
        assert "Introduction" in chunks[0].text
        assert "Intro text." in chunks[0].text
        assert "Setup" in chunks[1].text

    def test_rst_dash_underline(self, tmp_path: Path) -> None:
        rst_file = tmp_path / "guide.rst"
        rst_file.write_text(
            "Section A\n---------\nContent A.\n\nSection B\n---------\nContent B.\n"
        )
        chunks = chunk_file(rst_file, tmp_path)
        assert len(chunks) == 2

    def test_rst_tilde_underline(self, tmp_path: Path) -> None:
        rst_file = tmp_path / "guide.rst"
        rst_file.write_text("Part One\n~~~~~~~~\nText one.\n\nPart Two\n~~~~~~~~\nText two.\n")
        chunks = chunk_file(rst_file, tmp_path)
        assert len(chunks) == 2

    def test_rst_no_heading_fallback(self, tmp_path: Path) -> None:
        rst_file = tmp_path / "notes.rst"
        rst_file.write_text("Plain paragraph content.\nNo headings here.\n")
        chunks = chunk_file(rst_file, tmp_path)
        assert len(chunks) == 1


class TestPlainTextChunking:
    """Tests for plain text paragraph splitting."""

    def test_paragraph_splitting(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "readme.txt"
        # Each paragraph must exceed 2000 chars to avoid merging
        para_one = "First paragraph. " + "word " * 400
        para_two = "Second paragraph. " + "word " * 400
        para_three = "Third paragraph. " + "word " * 400
        txt_file.write_text(f"{para_one}\n\n{para_two}\n\n{para_three}\n")
        chunks = chunk_file(txt_file, tmp_path)
        assert len(chunks) == 3
        assert "First paragraph." in chunks[0].text
        assert "Second paragraph." in chunks[1].text
        assert "Third paragraph." in chunks[2].text

    def test_small_paragraphs_merged(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "readme.txt"
        # Many small paragraphs should be merged up to 2000 chars
        small_paragraphs = "\n\n".join(f"Paragraph {idx}." for idx in range(5))
        txt_file.write_text(small_paragraphs)
        chunks = chunk_file(txt_file, tmp_path)
        # All 5 small paragraphs should fit in one chunk (well under 2000 chars)
        assert len(chunks) == 1

    def test_large_paragraphs_stay_separate(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "readme.txt"
        # Two paragraphs each over 2000 chars
        large_para = "word " * 500  # ~2500 chars
        txt_file.write_text(f"{large_para}\n\n{large_para}")
        chunks = chunk_file(txt_file, tmp_path)
        assert len(chunks) == 2

    def test_unknown_extension_treated_as_plaintext(self, tmp_path: Path) -> None:
        other_file = tmp_path / "notes.log"
        other_file.write_text("First part.\n\nSecond part.\n")
        chunks = chunk_file(other_file, tmp_path)
        assert len(chunks) >= 1


class TestPathSafety:
    """Tests for symlink rejection, file size limits, extension filtering, total size limit."""

    def test_symlink_rejected(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.md"
        real_file.write_text("# Real content\n")
        link = tmp_path / "link.md"
        link.symlink_to(real_file)

        chunks = load_docs(tmp_path, extensions=[".md"])
        # Only the real file should be loaded, symlink skipped
        source_files = {chunk.source_file for chunk in chunks}
        assert "link.md" not in source_files
        assert "real.md" in source_files

    def test_file_size_limit(self, tmp_path: Path) -> None:
        small_file = tmp_path / "small.md"
        small_file.write_text("# Small\nContent.\n")
        big_file = tmp_path / "big.md"
        big_file.write_text("# Big\n" + "x" * (1024 * 1024 + 1))  # Over 1MB

        chunks = load_docs(tmp_path, extensions=[".md"])
        source_files = {chunk.source_file for chunk in chunks}
        assert "small.md" in source_files
        assert "big.md" not in source_files

    def test_extension_filter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        md_file.write_text("# Guide\nContent.\n")
        py_file = tmp_path / "script.py"
        py_file.write_text("# comment\nprint('hello')\n")

        chunks = load_docs(tmp_path, extensions=[".md"])
        source_files = {chunk.source_file for chunk in chunks}
        assert "guide.md" in source_files
        assert "script.py" not in source_files

    def test_total_size_limit(self, tmp_path: Path) -> None:
        # Create files that together exceed a small total limit
        for file_idx in range(10):
            file_path = tmp_path / f"doc{file_idx}.md"
            file_path.write_text(f"# Doc {file_idx}\n" + "content " * 100)

        # Use a very small total size limit
        chunks = load_docs(tmp_path, extensions=[".md"], max_total_size=500)
        # Should not load all files
        assert len(chunks) < 10

    def test_symlinked_directory_rejected(self, tmp_path: Path) -> None:
        """Files inside a symlinked directory should be skipped."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Real directory with a file, outside docs_dir
        real_dir = tmp_path / "outside"
        real_dir.mkdir()
        real_file = real_dir / "secret.md"
        real_file.write_text("# Secret\nHidden content.\n")

        # Symlink a directory inside docs_dir pointing to the real dir
        linked_dir = docs_dir / "linked"
        linked_dir.symlink_to(real_dir)

        # Also add a normal file inside docs_dir for comparison
        normal_file = docs_dir / "normal.md"
        normal_file.write_text("# Normal\nVisible content.\n")

        chunks = load_docs(docs_dir, extensions=[".md"])
        source_files = {chunk.source_file for chunk in chunks}
        assert "normal.md" in source_files
        # The file inside the symlinked directory should NOT appear
        assert all("secret.md" not in src for src in source_files)

    def test_path_traversal_guard(self, tmp_path: Path) -> None:
        # Create a file outside docs_root
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.md"
        outside_file.write_text("# Secret\nDon't read this.\n")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        # Create a symlink from docs to outside
        link_in_docs = docs_dir / "secret.md"
        link_in_docs.symlink_to(outside_file)

        chunks = load_docs(docs_dir, extensions=[".md"])
        source_files = {chunk.source_file for chunk in chunks}
        assert "secret.md" not in source_files


class TestDomainExtraction:
    """Tests for domain extraction from file paths."""

    def test_subdirectory_domain(self, tmp_path: Path) -> None:
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        setup_file = auth_dir / "setup.md"
        setup_file.write_text("# Auth Setup\nInstructions.\n")

        chunks = chunk_file(setup_file, tmp_path)
        assert chunks[0].domain == "auth"

    def test_root_level_general_domain(self, tmp_path: Path) -> None:
        readme = tmp_path / "readme.md"
        readme.write_text("# Readme\nGeneral info.\n")

        chunks = chunk_file(readme, tmp_path)
        assert chunks[0].domain == "general"

    def test_nested_subdirectory_uses_first_level(self, tmp_path: Path) -> None:
        nested = tmp_path / "api" / "v2" / "endpoints"
        nested.mkdir(parents=True)
        doc_file = nested / "users.md"
        doc_file.write_text("# Users API\nDetails.\n")

        chunks = chunk_file(doc_file, tmp_path)
        assert chunks[0].domain == "api"
