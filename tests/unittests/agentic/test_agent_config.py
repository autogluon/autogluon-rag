import os
import tempfile
import unittest

from agrag.args import Arguments


class TestAgentConfig(unittest.TestCase):
    def _write_config(self, contents: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        self.addCleanup(os.remove, path)
        return path

    def test_defaults_load_when_no_agent_section(self):
        # A config with no agent section falls back to the default yaml values.
        path = self._write_config("data:\n  data_dir: /tmp/data\n")
        args = Arguments(path)
        self.assertFalse(args.agent_enabled)
        self.assertEqual(args.agent_default_mode, "standard")
        self.assertEqual(args.agent_max_iterations, 5)
        self.assertEqual(args.agent_max_subqueries, 4)
        self.assertEqual(args.agent_retrieve_top_k_per_query, 8)
        self.assertEqual(args.agent_max_context_tokens, 6000)
        self.assertTrue(args.agent_use_query_rewrite)
        self.assertFalse(args.agent_use_context_compression)
        self.assertTrue(args.agent_use_verification)
        self.assertEqual(args.agent_min_evidence_count, 2)
        self.assertFalse(args.agent_return_trace)

    def test_user_values_override_defaults(self):
        path = self._write_config(
            "agent:\n"
            "  enabled: true\n"
            "  max_iterations: 5\n"
            "  use_verification: false\n"
            "  min_evidence_count: 1\n"
        )
        args = Arguments(path)
        self.assertTrue(args.agent_enabled)
        self.assertEqual(args.agent_max_iterations, 5)
        self.assertFalse(args.agent_use_verification)
        self.assertEqual(args.agent_min_evidence_count, 1)
        # Unspecified keys still come from defaults.
        self.assertEqual(args.agent_max_subqueries, 4)
        self.assertEqual(args.agent_default_mode, "standard")


if __name__ == "__main__":
    unittest.main()
