import importlib.util
import asyncio
from pathlib import Path


def test_frontend_examples_and_runtime_boundary():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"
    spec = importlib.util.spec_from_file_location("check_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def test_extension_has_five_namespaced_nodes_without_core_registration():
    import comfy.supported_models
    from llada_image_t8.nodes import LLaDAImageExtension
    from llada_image_t8.runtime.config import LLaDAImage

    node_classes = asyncio.run(LLaDAImageExtension().get_node_list())
    ids = [node.define_schema().node_id for node in node_classes]
    assert len(set(ids)) == len(ids) == 5
    assert all(node_id.startswith("T8") for node_id in ids)
    assert "T8LLaDAImageCheckpointLoader" in ids
    assert LLaDAImage not in comfy.supported_models.models


def test_core_loads_repository_named_directory():
    import nodes
    import comfy.supported_models

    package = Path(__file__).resolve().parents[1]
    before = tuple(comfy.supported_models.models)
    assert asyncio.run(nodes.load_custom_node(str(package)))
    assert tuple(comfy.supported_models.models) == before
    for name in (
        "T8LLaDAImageCheckpointLoader",
        "T8LLaDAImageVQConditioning",
    ):
        assert name in nodes.NODE_CLASS_MAPPINGS
