import torch
from comfy import latent_formats, supported_models_base, utils
from . import model_base, text_encoder


class LLaDAImage(supported_models_base.BASE):
    unet_config = {
        "image_model": "llada_image",
    }

    unet_extra_config = {}
    latent_format = latent_formats.Flux2
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 2.0
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        variant = unet_config["variant"]
        self.sampling_settings = {
            "multiplier": 1.0,
            "shift": 1.0 if variant == "base" else 3.0,
        }

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.LLaDAImage(self, device=device)

    def process_clip_state_dict(self, state_dict):
        state_dict = super().process_clip_state_dict(state_dict)
        state_dict = utils.state_dict_prefix_replace(
            state_dict,
            {
                "queryformer.": "llada2.queryformer.",
                "text_projection.": "llada2.text_projection.",
                "sigvq.": "llada2.sigvq.",
            },
        )
        suffixes = (
            ".mlp.experts.gate_proj",
            ".mlp.experts.up_proj",
            ".mlp.experts.down_proj",
        )
        for key in list(state_dict.keys()):
            if key.endswith(suffixes):
                state_dict[f"{key}.weight"] = state_dict.pop(key)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        state_dict = dict(state_dict)
        for key in list(state_dict.keys()):
            if key.endswith((
                ".mlp.experts.gate_proj.weight",
                ".mlp.experts.up_proj.weight",
                ".mlp.experts.down_proj.weight",
            )):
                state_dict[key[:-len(".weight")]] = state_dict.pop(key)
        return utils.state_dict_prefix_replace(
            state_dict,
            {
                "llada2.model.": "text_encoders.llada2.model.",
                "llada2.queryformer.": "text_encoders.queryformer.",
                "llada2.text_projection.": "text_encoders.text_projection.",
                "llada2.sigvq.": "text_encoders.sigvq.",
                "tokenizer_json": "text_encoders.tokenizer_json",
            },
            filter_keys=True,
        )

    def clip_target(self, state_dict={}):
        key = "text_encoders.llada2.model.language_model.word_embeddings.weight"
        if key not in state_dict:
            return None
        target = supported_models_base.ClipTarget(
            text_encoder.LLaDAImageTokenizer,
            text_encoder.te(dtype_llada=state_dict[key].dtype),
        )
        for key in (
            "llada2_config",
            "queryformer_config",
            "text_projection_config",
            "sigvq_config",
        ):
            if key in self.unet_config:
                target.params[key] = self.unet_config[key]
        return target
