"""AIO loading through ComfyUI's standard model, CLIP and VAE owners."""

import torch
from comfy import model_management, model_patcher, sd as comfy_sd, utils

from .config import LLaDAImage
from .detection import detect_config


def load_checkpoint(path, embedding_directory=None, component=None, disable_dynamic=False):
    state, metadata = utils.load_torch_file(path, return_metadata=True)
    prefix = "model.diffusion_model."
    config = LLaDAImage(detect_config(state, prefix, metadata))
    clip_target = config.clip_target(state)
    if clip_target is None or "text_encoders.tokenizer_json" not in state:
        raise ValueError("LLaDA-Image AIO requires embedded text encoder and tokenizer weights.")
    if not any(key.startswith("vae.") for key in state):
        raise ValueError("LLaDA-Image AIO requires embedded VAE weights.")

    parameters = utils.calculate_parameters(state, prefix)
    weight_dtype = utils.weight_dtype(state, prefix)
    config.quant_config = utils.detect_layer_quantization(state, prefix)
    if config.quant_config is not None:
        weight_dtype = None
    elif weight_dtype not in config.supported_inference_dtypes:
        raise ValueError("This release supports BF16/FP32 LLaDA-Image AIO checkpoints only.")
    load_device = model_management.get_torch_device()
    dtype = model_management.unet_dtype(
        model_params=parameters,
        supported_dtypes=config.supported_inference_dtypes,
        weight_dtype=weight_dtype,
    )
    manual_cast = model_management.unet_manual_cast(
        None if config.quant_config is not None else dtype,
        load_device, config.supported_inference_dtypes
    )
    config.set_inference_dtype(dtype, manual_cast, device=load_device)
    model = clip = vae = None
    if component in (None, "model"):
        initial_device = model_management.unet_inital_load_device(parameters, dtype)
        network = config.get_model(state, prefix, device=initial_device)
        patcher_class = model_patcher.ModelPatcher if disable_dynamic else model_patcher.CoreModelPatcher
        model = patcher_class(
            network, load_device=load_device,
            offload_device=model_management.unet_offload_device(),
        )
        network.load_model_weights(state, prefix, assign=model.is_dynamic())
        model.cached_patcher_init = (load_checkpoint, (path, embedding_directory, "model"), 0)

    if component in (None, "vae"):
        vae_state = utils.state_dict_prefix_replace(state, {"vae.": ""}, filter_keys=True)
        vae = comfy_sd.VAE(sd=config.process_vae_state_dict(vae_state), metadata=metadata)
        vae.patcher.cached_patcher_init = (reload_vae_patcher, (path, embedding_directory))
    if component in (None, "clip"):
        clip_state = config.process_clip_state_dict(state)
        clip = comfy_sd.CLIP(
            clip_target, embedding_directory=embedding_directory,
            tokenizer_data=clip_state, parameters=utils.calculate_parameters(clip_state),
            state_dict=clip_state, disable_dynamic=disable_dynamic,
        )
        clip.patcher.cached_patcher_init = (reload_clip_patcher, (path, embedding_directory))
    if model is not None and initial_device != torch.device("cpu"):
        model_management.load_models_gpu([model], force_full_load=True)
    return model, clip, vae


def reload_clip_patcher(path, embedding_directory=None, disable_dynamic=False):
    return load_checkpoint(path, embedding_directory, "clip", disable_dynamic)[1].patcher


def reload_vae_patcher(path, embedding_directory=None, disable_dynamic=False):
    return load_checkpoint(path, embedding_directory, "vae", disable_dynamic)[2].patcher
