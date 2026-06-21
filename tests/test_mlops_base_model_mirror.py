from __future__ import annotations

import unittest

from sqlbench_lab.mlops import (
    DEV_BASE_MODEL_MIRROR_SCHEMA_VERSION,
    build_dev_base_model_mirror_plan,
)


class SQLAdapterBaseModelMirrorTests(unittest.TestCase):
    def test_base_model_mirror_plan_names_hf_revision_and_gcs_prefix(self) -> None:
        plan = build_dev_base_model_mirror_plan(
            base_model="Qwen/Qwen3.5-0.8B-Base",
            revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            local_dir="artifacts/dev/base_models/qwen35",
        )

        self.assertEqual(plan.schema_version, DEV_BASE_MODEL_MIRROR_SCHEMA_VERSION)
        self.assertEqual(plan.base_model, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(plan.revision, "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
        self.assertEqual(
            plan.gcs_uri,
            (
                "gs://mistri-sqlbench-dev-models/base-models/"
                "Qwen_Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/"
            ),
        )
        self.assertEqual(plan.download_command[:4], ("huggingface-cli", "download", "Qwen/Qwen3.5-0.8B-Base", "--revision"))
        self.assertEqual(plan.upload_command[:4], ("gsutil", "-m", "rsync", "-r"))

    def test_base_model_mirror_plan_requires_gcs_uri(self) -> None:
        with self.assertRaisesRegex(ValueError, "gcs_uri must start with gs://"):
            build_dev_base_model_mirror_plan(
                base_model="Qwen/Qwen3.5-0.8B-Base",
                revision="main",
                local_dir="artifacts/dev/base_models/qwen35",
                gcs_uri="s3://bucket/model",
            )


if __name__ == "__main__":
    unittest.main()
