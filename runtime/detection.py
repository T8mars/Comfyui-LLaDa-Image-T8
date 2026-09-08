import json
from collections.abc import Mapping

from comfy.model_detection import count_blocks


def detect_config(state_dict, key_prefix="model.diffusion_model.", metadata=None):
    state_dict_keys = state_dict.keys()
    llada_x_key = f"{key_prefix}all_x_embedder.1-1.weight"
    llada_noise_key = f"{key_prefix}noise_refiner.0.attention.to_q.weight"
    llada_sigvq_key = f"{key_prefix}sigvq_refiner.0.attention.to_q.weight"
    if llada_x_key in state_dict_keys and llada_noise_key in state_dict_keys and llada_sigvq_key in state_dict_keys:
        checkpoint_config = {}
        if metadata is not None and "config" in metadata:
            checkpoint_config = json.loads(metadata["config"])
        transformer_config = checkpoint_config.get("transformer", {})
        llada_config = checkpoint_config.get("llada_image", {})

        def component_config(name):
            config = checkpoint_config.get(name)
            if config is not None and not isinstance(config, Mapping):
                raise ValueError(f"LLaDA-Image {name} config must be a mapping")
            return config

        component_configs = {
            "llada2_config": component_config("text_encoder"),
            "queryformer_config": component_config("queryformer"),
            "text_projection_config": component_config("text_projection"),
            "sigvq_config": component_config("sigvq"),
        }
        text_projection_config = component_configs["text_projection_config"]
        sigvq_config = component_configs["sigvq_config"]

        def component_dimension(config, component, key, default):
            if config is None:
                return default
            value = config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"LLaDA-Image {component}.{key} must be a positive integer"
                )
            return value

        default_cap_feat_dim = component_dimension(
            text_projection_config, "text_projection", "projection_dim", 2560
        )
        default_semantic_feat_dim = component_dimension(
            sigvq_config, "sigvq", "semantic_embed_dim", 4096
        )
        variant = llada_config.get("variant")
        if variant not in ("base", "turbo"):
            raise ValueError("LLaDA-Image AIO checkpoint metadata must identify the base or turbo variant")

        final_weight = state_dict.get(f"{key_prefix}all_final_layer.1-1.linear.weight")
        if final_weight is None:
            raise ValueError("LLaDA-Image checkpoint is missing all_final_layer.1-1.linear.weight")
        detected_dim = state_dict[llada_x_key].shape[0]
        detected_layers = count_blocks(state_dict_keys, f"{key_prefix}layers.{{}}.")
        detected_refiners = count_blocks(state_dict_keys, f"{key_prefix}noise_refiner.{{}}.")
        if count_blocks(state_dict_keys, f"{key_prefix}context_refiner.{{}}.") != detected_refiners:
            raise ValueError("LLaDA-Image context and noise refiner layer counts differ")
        if count_blocks(state_dict_keys, f"{key_prefix}sigvq_refiner.{{}}.") != detected_refiners:
            raise ValueError("LLaDA-Image SigVQ and noise refiner layer counts differ")
        inferred_values = {
            "in_channels": final_weight.shape[0],
            "dim": detected_dim,
            "n_layers": detected_layers,
            "n_refiner_layers": detected_refiners,
        }
        if inferred_values["in_channels"] != 128:
            raise ValueError(
                "LLaDA-Image requires 128 transformer input channels for the "
                "Flux2 latent format"
            )
        for name, value in inferred_values.items():
            if name in transformer_config and transformer_config[name] != value:
                raise ValueError(
                    f"LLaDA-Image transformer config declares {name}={transformer_config[name]!r}, "
                    f"but the state dict implies {value!r}"
                )
        dit_config = {
            "image_model": "llada_image",
            "variant": variant,
            "all_patch_size": tuple(transformer_config.get("all_patch_size", [1])),
            "all_f_patch_size": tuple(transformer_config.get("all_f_patch_size", [1])),
            "in_channels": inferred_values["in_channels"],
            "dim": inferred_values["dim"],
            "n_layers": inferred_values["n_layers"],
            "n_refiner_layers": inferred_values["n_refiner_layers"],
            "n_heads": transformer_config.get("n_heads", 30),
            "norm_eps": transformer_config.get("norm_eps", 1e-5),
            "qk_norm": transformer_config.get("qk_norm", True),
            "cap_feat_dim": transformer_config.get("cap_feat_dim", default_cap_feat_dim),
            "semantic_feat_dim": transformer_config.get(
                "semantic_feat_dim", default_semantic_feat_dim
            ),
            "rope_theta": transformer_config.get("rope_theta", 256.0),
            "t_scale": transformer_config.get("t_scale", 1000.0),
            "axes_dims": tuple(transformer_config.get("axes_dims", [32, 48, 48])),
        }
        if dit_config["dim"] % dit_config["n_heads"]:
            raise ValueError("LLaDA-Image hidden dimension must be divisible by its attention heads")
        if sum(dit_config["axes_dims"]) != dit_config["dim"] // dit_config["n_heads"]:
            raise ValueError("LLaDA-Image RoPE axis dimensions must sum to the attention head dimension")
        text_config = component_configs["llada2_config"]
        queryformer_config = component_configs["queryformer_config"]

        def require_matching_config(
            left_name, left_config, left_key, right_name, right_config, right_key
        ):
            if left_config is None or right_config is None:
                return
            left = left_config.get(left_key)
            right = right_config.get(right_key)
            if left is not None and right is not None and left != right:
                raise ValueError(
                    f"LLaDA-Image config mismatch: {left_name}.{left_key}={left!r}, "
                    f"but {right_name}.{right_key}={right!r}"
                )

        require_matching_config(
            "text_encoder",
            text_config,
            "hidden_size",
            "queryformer",
            queryformer_config,
            "hidden_size",
        )
        require_matching_config(
            "text_encoder",
            text_config,
            "hidden_size",
            "text_projection",
            text_projection_config,
            "hidden_size",
        )
        require_matching_config(
            "transformer",
            dit_config,
            "cap_feat_dim",
            "text_projection",
            text_projection_config,
            "projection_dim",
        )
        require_matching_config(
            "transformer",
            dit_config,
            "semantic_feat_dim",
            "sigvq",
            sigvq_config,
            "semantic_embed_dim",
        )
        dit_config.update({key: value for key, value in component_configs.items() if value is not None})
        return dit_config

    raise ValueError("Select a converted LLaDA-Image Base or Turbo AIO safetensors checkpoint.")
