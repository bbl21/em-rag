import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _dep_name(spec: str) -> str:
    base = spec.strip().split("[", 1)[0]
    return re.split(r"[<=>!~ ]", base, 1)[0].strip()


class BootstrapContainerContractTests(unittest.TestCase):
    def test_pyproject_build_dependency_group_contains_parser_deps(self) -> None:
        project_file = ROOT / "pyproject.toml"
        with project_file.open("rb") as fp:
            project = tomllib.load(fp)

        build_deps = project.get("project", {}).get("optional-dependencies", {}).get("build", [])
        build_names = {_dep_name(item).lower() for item in build_deps}

        self.assertIn("pyyaml", build_names)
        self.assertIn("pypdf", build_names)
        self.assertIn("pymupdf", build_names)

    def test_runtime_requirements_includes_parser_deps(self) -> None:
        req_file = ROOT / "requirements-runtime.txt"
        lines = [
            line
            for line in req_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        names = {_dep_name(line).lower() for line in lines}

        self.assertIn("pyyaml", names)
        self.assertIn("pypdf", names)
        self.assertIn("pymupdf", names)

    def test_dockerfile_uses_build_extras_and_non_root_uid(self) -> None:
        docker_file = ROOT / "Dockerfile"
        text = docker_file.read_text(encoding="utf-8")

        self.assertIn("python -m pip install --no-cache-dir .[runtime,build]", text)
        self.assertIn("COPY demo ./demo", text)
        self.assertIn("useradd --create-home --uid 1000 emrag", text)
        self.assertNotIn("--uid 10001", text)

    def test_compose_uses_uid_gid_runtime_user_mapping(self) -> None:
        compose_file = ROOT / "compose.yaml"
        text = compose_file.read_text(encoding="utf-8")

        mapping_line = 'user: "${EM_RAG_DOCKER_UID:-1000}:${EM_RAG_DOCKER_GID:-1000}"'
        self.assertIn(mapping_line, text)
        self.assertIn("EM_RAG_DOCKER_UID: \"${EM_RAG_DOCKER_UID:-1000}\"", text)
        self.assertIn("EM_RAG_DOCKER_GID: \"${EM_RAG_DOCKER_GID:-1000}\"", text)
        self.assertIn("- ./kb_corpus_build:/app/kb_corpus_build", text)
        self.assertIn('profiles: ["rag-download"]', text)
        self.assertIn('command: ["python", "-m", "em_rag.bootstrap", "acquire-sources", "--project-root", "/app"]', text)
        self.assertIn("- ./reference:/app/reference:ro", text)
        self.assertIn("- ./reference:/app/reference\n", text)

    def test_bootstrap_source_release_readme_contract(self) -> None:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")

        self.assertIn('.[runtime,test]', text)
        self.assertIn("prepare-model", text)
        self.assertIn("bootstrap demo", text)
        self.assertIn("NEEDS_CALIBRATION", text)
        self.assertIn("optional development material", text)
