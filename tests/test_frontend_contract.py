"""Static API-contract checks for the browser client without adding a JS runtime."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_frontend_reads_score_and_never_renders_rejected_candidates(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("candidate.score", script)
        self.assertIn("if (!result.recognized)", script)
        self.assertIn("result.candidates.length !== 3", script)
        self.assertNotIn("candidate.probability", script)

    def test_frontend_keeps_unicode_latex_copy_and_processed_preview_contracts(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn("candidate.unicode", script)
        self.assertIn("candidate.latex", script)
        self.assertIn("navigator.clipboard.writeText(candidate.latex)", script)
        self.assertIn("include_preview: previewToggle.checked", script)
        self.assertIn('id="processed-preview-image"', template)


if __name__ == "__main__":
    unittest.main()
