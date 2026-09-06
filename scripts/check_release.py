"""Offline release checks; shipped examples must be frontend graphs, not prompts."""

import ast
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {f"{variant}_{mode}.json" for variant in ("base", "turbo") for mode in ("text", "vq", "editing")}


def check_workflow(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(graph.get("nodes"), list), f"{path.name}: not a frontend workflow"
    assert isinstance(graph.get("links"), list), f"{path.name}: missing UI links"
    assert graph.get("version") == 0.4, f"{path.name}: unsupported workflow format"
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"]), f"{path.name}: duplicate node IDs"
    assert any(node["type"] == "T8LLaDAImageCheckpointLoader" for node in nodes.values())
    assert not any(node["type"] == "CheckpointLoaderSimple" for node in nodes.values())
    links = {link[0]: link for link in graph["links"]}
    assert len(links) == len(graph["links"]), f"{path.name}: duplicate links"
    for link_id, source, source_slot, target, target_slot, kind in graph["links"]:
        output = nodes[source]["outputs"][source_slot]
        input_ = nodes[target]["inputs"][target_slot]
        assert link_id in output["links"], f"{path.name}: disconnected output {link_id}"
        assert input_["link"] == link_id, f"{path.name}: disconnected input {link_id}"
        assert output["type"] == input_["type"] == kind
    for node in nodes.values():
        assert len(node["pos"]) == len(node["size"]) == 2
        for input_ in node.get("inputs", []):
            assert input_.get("link") is None or input_["link"] in links
        if node["type"].startswith("T8"):
            assert node["properties"]["cnr_id"] == "llada-image-t8"
    return len(nodes), len(links)


def main(require_acceptance=False):
    examples = ROOT / "example_workflows"
    assert {path.name for path in examples.glob("*.json")} == EXPECTED
    for path in sorted(examples.glob("*.json")):
        count, links = check_workflow(path)
        print(f"{path.name}: frontend graph, {count} nodes, {links} links")
    for path in [ROOT / "nodes.py", *sorted((ROOT / "runtime").glob("*.py"))]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for item in ast.walk(tree):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                assert item.func.id not in {"eval", "exec"}, str(path)
        assert "comfy.ldm.llada_image" not in source
        assert "comfy.text_encoders.llada_image" not in source
    print("Standalone import boundary and frontend workflow checks passed.")
    if require_acceptance:
        evidence = json.loads((ROOT / "docs" / "acceptance.json").read_text(encoding="utf-8"))
        assert evidence["status"] == "passed"
        assert {item["workflow"] for item in evidence["results"]} == EXPECTED
        assert len(evidence["results"]) == 6
        for item in evidence["results"]:
            assert item["status"] == "success"
            assert item["saved_via_frontend"]
            assert item["native_baseline_pixel_exact"]
            path = examples / item["workflow"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["workflow_sha256"], path.name
        for name, digest in evidence["runtime_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        assert evidence["tests"]["failed"] == evidence["tests"]["skipped"] == 0
        assert evidence["tests"]["passed"] >= 123
        print("Six completed frontend executions and release file hashes verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-acceptance", action="store_true")
    main(parser.parse_args().require_acceptance)
