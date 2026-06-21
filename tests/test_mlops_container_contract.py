from __future__ import annotations

import unittest
from pathlib import Path

from sqlbench_lab.mlops.container_contract import (
    DEV_CLI_IMAGE_NAME,
    DEV_CLI_IMAGE_TAG,
    DEV_CONTAINER_DOCKERFILE,
    build_dev_cli_docker_build_command,
    build_dev_cli_docker_run_command,
    dev_cli_container_contract,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SQLAdapterContainerContractTests(unittest.TestCase):
    def test_dev_cli_container_contract_names_image_and_commands(self) -> None:
        contract = dev_cli_container_contract()

        self.assertEqual(contract.image_name, DEV_CLI_IMAGE_NAME)
        self.assertEqual(contract.image_tag, DEV_CLI_IMAGE_TAG)
        self.assertEqual(contract.dockerfile_path, DEV_CONTAINER_DOCKERFILE)
        self.assertEqual(contract.entrypoint, ("python", "-m", "sqlbench_lab.cli"))
        self.assertIn("sql validate-manifest", contract.supported_command_summaries)
        self.assertIn("sql run-sft --dry-run", contract.supported_command_summaries)
        self.assertIn("sql eval --openai-base-url", contract.supported_command_summaries)
        self.assertIn("sql openai-load-test", contract.supported_command_summaries)
        self.assertEqual(contract.default_dependency_groups, ())
        self.assertEqual(contract.optional_dependency_groups, ("mlops", "training", "serving"))

    def test_docker_commands_are_stable(self) -> None:
        self.assertEqual(
            build_dev_cli_docker_build_command(tag="sqlbench-lab-dev-cli:test"),
            (
                "docker",
                "build",
                "-f",
                "docker/sqlbench-dev-cli.Dockerfile",
                "--build-arg",
                "INSTALL_GROUPS=mlops",
                "-t",
                "sqlbench-lab-dev-cli:test",
                ".",
            ),
        )
        self.assertEqual(
            build_dev_cli_docker_run_command(
                tag="sqlbench-lab-dev-cli:test",
                cli_args=(
                    "sql",
                    "validate-manifest",
                    "--manifest",
                    "experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json",
                ),
            ),
            (
                "docker",
                "run",
                "--rm",
                "sqlbench-lab-dev-cli:test",
                "sql",
                "validate-manifest",
                "--manifest",
                "experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json",
            ),
        )

    def test_dockerfile_and_context_ignore_exist(self) -> None:
        dockerfile = WORKSPACE_ROOT / DEV_CONTAINER_DOCKERFILE
        dockerignore = WORKSPACE_ROOT / ".dockerignore"

        self.assertTrue(dockerfile.exists())
        self.assertTrue(dockerignore.exists())
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        dockerignore_text = dockerignore.read_text(encoding="utf-8")
        self.assertIn("ENTRYPOINT", dockerfile_text)
        self.assertIn("python", dockerfile_text)
        self.assertIn("-m", dockerfile_text)
        self.assertIn("sqlbench_lab.cli", dockerfile_text)
        self.assertIn("INSTALL_GROUPS", dockerfile_text)
        self.assertIn("uv sync --frozen --no-dev", dockerfile_text)
        self.assertIn(".venv", dockerignore_text)
        self.assertIn("artifacts/", dockerignore_text)
        self.assertIn("results/", dockerignore_text)
        self.assertIn(".metaflow/", dockerignore_text)


if __name__ == "__main__":
    unittest.main()
