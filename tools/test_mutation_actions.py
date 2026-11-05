import copy
import json
from pathlib import Path
import tempfile
import unittest

from merge_mutation_shards import file_digest
from mutation_actions import checked_manifest, merge_directory, plan_outputs
from test_merge_mutation_shards import campaign


class ActionsMutationContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest, self.results = campaign()
        self.revision = self.manifest["source"]["revision"]
        self.path = self.root / "campaign.json"
        self.path.write_text(json.dumps(self.manifest))
        self.sha256 = file_digest(self.path)

    def test_publish_pins_exact_file_and_all_zero_based_shards(self):
        outputs = plan_outputs(self.path, self.revision)
        self.assertEqual(outputs["manifest_sha256"], self.sha256)
        self.assertNotEqual(outputs["manifest_sha256"], self.manifest["manifest_sha256"])
        self.assertEqual(json.loads(outputs["matrix"]), {"include": [
            {"index": 0, "shard": "0/3"}, {"index": 1, "shard": "1/3"},
            {"index": 2, "shard": "2/3"}]})

    def test_independent_digest_rejects_even_semantically_identical_replacement(self):
        self.path.write_text(json.dumps(self.manifest, indent=2))
        with self.assertRaisesRegex(ValueError, "prepare job digest"):
            checked_manifest(self.path, self.revision, self.sha256)

    def test_manifest_cannot_select_another_workflow_revision(self):
        with self.assertRaisesRegex(ValueError, "workflow snapshot"):
            checked_manifest(self.path, "9" * 40, self.sha256)

    def test_aggregate_preserves_complete_shards_and_raw_outcomes(self):
        for index, result in enumerate(self.results):
            directory = self.root / "downloads" / f"artifact-{index}" / "target/quality"
            directory.mkdir(parents=True)
            (directory / f"shard-result-{index}.json").write_text(json.dumps(result))
        output = self.root / "aggregate.json"
        manifest = checked_manifest(self.path, self.revision, self.sha256)
        merge_directory(manifest, self.root / "downloads", output)
        aggregate = json.loads(output.read_text())
        self.assertEqual(aggregate["total_mutants"], len(self.manifest["candidates"]))
        self.assertEqual(len(aggregate["shard_baselines"]), 3)
        self.assertEqual(aggregate["campaign_manifest"], self.manifest)

    def test_partial_artifact_set_never_publishes_aggregate(self):
        (self.root / "shard-result-0.json").write_text(json.dumps(self.results[0]))
        output = self.root / "aggregate.json"
        with self.assertRaisesRegex(ValueError, "missing or extra shard"):
            merge_directory(self.manifest, self.root, output)
        self.assertFalse(output.exists())

    def test_old_attempt_duplicate_cannot_satisfy_missing_shard(self):
        for index in range(3):
            result = copy.deepcopy(self.results[index if index < 2 else 0])
            (self.root / f"shard-result-{index}.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "duplicate shard"):
            merge_directory(self.manifest, self.root, self.root / "aggregate.json")


if __name__ == "__main__":
    unittest.main()
