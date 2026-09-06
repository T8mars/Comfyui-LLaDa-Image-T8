import argparse
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import convert_comfyui_aio as converter


class ConvertComfyUIAIOTests(unittest.TestCase):
    def write_safetensors(self, path: Path, key: str, values: list[float]) -> bytes:
        data = struct.pack(f"<{len(values)}f", *values)
        header = json.dumps(
            {
                key: {
                    "dtype": "F32",
                    "shape": [len(values)],
                    "data_offsets": [0, len(data)],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + data)
        return data

    def make_source(self, root: Path, variant: str = "base") -> dict[str, bytes]:
        expected = {}
        source_keys = {
            "transformer": "x_pad_token",
            "text_encoder": "model.weight",
            "queryformer": "meta_queries",
            "text_projection": "projector.weight",
            "sigvq": "prior_token_embedding.weight",
            "vae": "quant_conv.weight",
        }
        for index, (component, prefix) in enumerate(converter.COMPONENT_PREFIXES):
            directory = root / component
            directory.mkdir(parents=True)
            values = [float(value + index) for value in range(index + 1)]
            source_key = source_keys[component]
            data = self.write_safetensors(
                directory / "model.safetensors", source_key, values
            )
            (directory / "config.json").write_text(
                json.dumps({"component": component}), encoding="utf-8"
            )
            expected[f"{prefix}{source_key}"] = data

        scheduler = {
            "shift": 1.0 if variant == "base" else 3.0,
            "stochastic_sampling": variant == "turbo",
        }
        if variant == "turbo":
            scheduler["use_uniform_sigmas"] = True
        scheduler_directory = root / "scheduler"
        scheduler_directory.mkdir()
        (scheduler_directory / "scheduler_config.json").write_text(
            json.dumps(scheduler), encoding="utf-8"
        )

        tokenizer_directory = root / "tokenizer"
        tokenizer_directory.mkdir()
        tokenizer_data = (
            b'{"version":"1.0","model":{"type":"WordLevel","vocab":{"hello":0}}}'
        )
        (tokenizer_directory / "tokenizer.json").write_bytes(tokenizer_data)
        (tokenizer_directory / "tokenizer_config.json").write_text(
            "{}", encoding="utf-8"
        )
        (root / "model_index.json").write_text("{}", encoding="utf-8")
        expected[converter.TOKENIZER_KEY] = tokenizer_data
        return expected

    def args(
        self, root: Path, output: Path, variant: str = "base"
    ) -> argparse.Namespace:
        source_files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if (
                relative.endswith((".safetensors", ".safetensors.index.json"))
                or relative == "model_index.json"
                or relative == "scheduler/scheduler_config.json"
                or relative
                in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
                or (relative.count("/") == 1 and relative.endswith("/config.json"))
            ):
                source_files.append(
                    {
                        "path": relative,
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        source_lock = root.parent / f"{variant}.source.json"
        source_lock.write_text(
            json.dumps(
                {
                    "format_version": converter.FORMAT_VERSION,
                    "variant": variant,
                    "source_repo": "inclusionAI/LLaDA-Image",
                    "source_revision": "0123456789abcdef",
                    "files": sorted(source_files, key=lambda entry: entry["path"]),
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            input=root,
            output=output,
            variant=variant,
            source_repo="inclusionAI/LLaDA-Image",
            source_revision="0123456789abcdef",
            source_lock=source_lock,
            overwrite=False,
        )

    def test_atomic_json_write_fsyncs_and_replaces(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            path.write_text('{"old":true}', encoding="utf-8")

            with patch.object(
                converter.os, "fsync", wraps=converter.os.fsync
            ) as fsync:
                converter.write_json_atomic(path, {"value": "完整"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"value": "完整"}
            )
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(fsync.call_count, 1)
            self.assertFalse(path.with_name(f"{path.name}.tmp").exists())

    def test_conversion_preserves_tensors_and_embeds_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            expected = self.make_source(root)
            output = Path(temporary_directory) / "llada-image-base.safetensors"

            converter.convert(self.args(root, output))

            header, data_start = converter.read_safetensors_header(output)
            self.assertEqual(set(header) - {"__metadata__"}, set(expected))
            with output.open("rb") as checkpoint:
                for key, value in expected.items():
                    start, end = header[key]["data_offsets"]
                    checkpoint.seek(data_start + start)
                    self.assertEqual(checkpoint.read(end - start), value, key)
            metadata = header["__metadata__"]
            config = json.loads(metadata["config"])
            self.assertEqual(config["llada_image"]["variant"], "base")
            self.assertEqual(
                metadata["llada_image.source_revision"], "0123456789abcdef"
            )

            manifest_path = output.with_suffix(".safetensors.manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tensor_count"], len(expected))
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertTrue(all(source["sha256"] for source in manifest["sources"]))
            source_lock = json.loads(
                self.args(root, output).source_lock.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["llada_image.source_lock_sha256"],
                converter.canonical_json_sha256(source_lock),
            )

    def test_variant_mismatch_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root, variant="base")
            output = Path(temporary_directory) / "bad.safetensors"

            with self.assertRaisesRegex(ValueError, "does not match"):
                converter.convert(self.args(root, output, variant="turbo"))
            self.assertFalse(output.exists())

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root)
            output = Path(temporary_directory) / "existing.safetensors"
            output.write_bytes(b"keep")

            with self.assertRaises(FileExistsError):
                converter.convert(self.args(root, output))
            self.assertEqual(output.read_bytes(), b"keep")

    def test_local_verification_failure_never_installs_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root)
            output = Path(temporary_directory) / "unverified.safetensors"

            with (
                patch.object(
                    converter,
                    "verify_checkpoint",
                    side_effect=ValueError("verification failed"),
                ),
                self.assertRaisesRegex(ValueError, "verification failed"),
            ):
                converter.convert(self.args(root, output))

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(f"{output.name}.partial").exists())

    def test_local_recovers_verified_output_when_manifest_was_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root)
            output = Path(temporary_directory) / "manifest-recovery.safetensors"
            args = self.args(root, output)

            with (
                patch.object(
                    converter,
                    "build_manifest",
                    side_effect=RuntimeError("manifest interrupted"),
                ),
                self.assertRaisesRegex(RuntimeError, "manifest interrupted"),
            ):
                converter.convert(args)

            state_path = converter.verified_state_path(output)
            self.assertTrue(output.exists())
            self.assertTrue(state_path.exists())
            self.assertFalse(output.with_name(f"{output.name}.partial").exists())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                state["verified_output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )

            with patch.object(
                converter,
                "write_checkpoint",
                side_effect=AssertionError("recovery rewrote tensor payloads"),
            ):
                converter.convert(args)

            manifest_path = output.with_suffix(".safetensors.manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertFalse(state_path.exists())

    def test_unknown_component_key_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root)
            self.write_safetensors(
                root / "queryformer" / "model.safetensors", "unknown.weight", [1.0]
            )

            with self.assertRaisesRegex(ValueError, "unknown queryformer tensor key"):
                converter.convert(
                    self.args(root, Path(temporary_directory) / "bad.safetensors")
                )

    def test_tensor_size_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.safetensors"
            header = json.dumps(
                {"x_pad_token": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
                separators=(",", ":"),
            ).encode("utf-8")
            header += b" " * ((-len(header)) % 8)
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            root = path.parent / "source"
            (root / "transformer").mkdir(parents=True)
            path.replace(root / "transformer" / "model.safetensors")

            with self.assertRaisesRegex(ValueError, "byte size"):
                converter.collect_component(
                    root, "transformer", "model.diffusion_model."
                )

    def test_unmapped_tensor_data_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            directory = root / "transformer"
            directory.mkdir(parents=True)
            self.write_safetensors(
                directory / "model.safetensors", "x_pad_token", [1.0]
            )
            with (directory / "model.safetensors").open("ab") as handle:
                handle.write(b"extra")

            with self.assertRaisesRegex(ValueError, "trailing tensor-data bytes"):
                converter.collect_component(
                    root, "transformer", "model.diffusion_model."
                )

    def test_source_lock_hash_mismatch_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            self.make_source(root)
            output = Path(temporary_directory) / "bad.safetensors"
            args = self.args(root, output)
            config_path = root / "transformer" / "config.json"
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "transformer", "transformez"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                converter.convert(args)
            self.assertFalse(output.exists())

    def test_local_conversion_uses_the_shared_output_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "source"
            root.mkdir()
            output = Path(temporary_directory) / "locked.safetensors"
            args = argparse.Namespace(input=root, output=output)

            with (
                converter.output_lock(output),
                self.assertRaisesRegex(RuntimeError, "already using output"),
            ):
                converter.convert(args)
