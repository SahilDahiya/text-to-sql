from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlbench_lab.docs_site import build_docs_site

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class DocsSiteTests(unittest.TestCase):
    def test_build_docs_site_writes_dense_browser_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "site"

            summary = build_docs_site(output_dir)

            index_html = (output_dir / "index.html").read_text(encoding="utf-8")
            training_html = (output_dir / "training.html").read_text(encoding="utf-8")
            learnings_html = (output_dir / "learnings.html").read_text(encoding="utf-8")
            research_html = (output_dir / "research.html").read_text(encoding="utf-8")
            runbook_html = (output_dir / "runbook.html").read_text(encoding="utf-8")
            observability_html = (output_dir / "observability.html").read_text(encoding="utf-8")
            documentation_html = (output_dir / "documentation.html").read_text(encoding="utf-8")
            experiment_html = (
                output_dir
                / "experiments"
                / "qwen35_0_8b__exp028_trl_regional_sales_unit_price_slot_v3.html"
            ).read_text(encoding="utf-8")

        self.assertGreaterEqual(summary.page_count, 9)
        self.assertEqual(summary.asset_count, 1)
        self.assertIn("SQLBench Lab", index_html)
        self.assertIn("System Map", index_html)
        self.assertIn("Exp031", index_html)
        self.assertIn("restaurant plus airline is prompt-dev", index_html)
        self.assertIn("works_cycles plus public_review_platform", index_html)
        self.assertIn("7/50", index_html)
        self.assertIn("Recent Experiment Ledger", training_html)
        self.assertIn("Exp028", training_html)
        self.assertIn("Practical fine-tuning lessons", learnings_html)
        self.assertIn("Canonical slot placement worked", learnings_html)
        self.assertIn("Exp029", learnings_html)
        self.assertIn("Exp030", learnings_html)
        self.assertIn("Exp031", learnings_html)
        self.assertIn("Exp032 plan", learnings_html)
        self.assertIn("Exp032 manual candidates", learnings_html)
        self.assertIn("Exp033 setup", learnings_html)
        self.assertIn("Blacksmith reset", learnings_html)
        self.assertIn("c000_current_system", learnings_html)
        self.assertIn("c001_schema_grounded", learnings_html)
        self.assertIn("every candidate is tracked in MLflow", learnings_html)
        self.assertIn("Expansion alone is not enough", learnings_html)
        self.assertIn("Text-to-SQL Research Map", research_html)
        self.assertIn("Blacksmith Moves", research_html)
        self.assertIn("Automatic Metadata Extraction", research_html)
        self.assertIn("Candidate Pool + Execution + Selection", research_html)
        self.assertIn("The Death of Schema Linking?", research_html)
        self.assertIn("Modern Fine-Tuning Pipeline", research_html)
        self.assertIn("Training Hygiene Rules", research_html)
        self.assertIn("Commands are documented", runbook_html)
        self.assertIn("Optimize prompt candidate", runbook_html)
        self.assertIn("Report token lengths", runbook_html)
        self.assertIn("Evaluate candidate pool", runbook_html)
        self.assertIn("Every MIPROv2/GEPA candidate has MLflow tags", runbook_html)
        self.assertIn("optimizer</span>", observability_html)
        self.assertIn("prompt_dev_dataset", observability_html)
        self.assertIn("Do not add markdown docs", documentation_html)
        self.assertIn("Train Inputs", experiment_html)
        self.assertIn("Eval Results", experiment_html)

    def test_repo_docs_do_not_use_markdown_docs(self) -> None:
        docs_dir = WORKSPACE_ROOT / "docs"
        markdown_docs = sorted(path.name for path in docs_dir.glob("*.md"))

        self.assertEqual(markdown_docs, [])
