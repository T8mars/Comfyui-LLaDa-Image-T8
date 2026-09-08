import torch
import comfy.conds
import comfy.model_sampling
from comfy.model_base import BaseModel, ModelType
from . import model


class LLaDAImageSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST
):
    def __init__(self, model_config, inference_dtype):
        super().__init__(model_config)
        self.inference_dtype = inference_dtype

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # The reference creates full-strength FP32 noise, quantizes it through the
        # transformer dtype, then returns it to FP32 before the first model call.
        noise = noise.to(dtype=self.inference_dtype).to(dtype=torch.float32)
        return noise + latent_image

    def inpaint_noise_scaling(self, sigma, noise, latent_image):
        noise = noise.to(dtype=self.inference_dtype).to(dtype=torch.float32)
        return super().noise_scaling(sigma, noise, latent_image)


class LLaDAImage(BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(
            model_config,
            ModelType.FLOW,
            device=device,
            unet_model=model.LLaDAImage,
        )
        self.model_sampling = LLaDAImageSampling(
            model_config, self.get_dtype_inference()
        )
        self.model_sampling.llada_image_variant = model_config.unet_config["variant"]
        self.memory_usage_factor_conds = ("source_latents", "semantic_features")

    def process_timestep(self, timestep, **kwargs):
        # The upstream pipeline rounds t before the transformer's t_scale multiply.
        return timestep.to(dtype=self.get_dtype_inference())

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1))
        return self.model_sampling.inpaint_noise_scaling(sigma, noise, latent_image)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn")
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            out["attention_mask"] = comfy.conds.CONDRegular(attention_mask)
        semantic_features = kwargs.get("semantic_features")
        if semantic_features is not None:
            out["semantic_features"] = comfy.conds.CONDRegular(semantic_features)
        semantic_mask = kwargs.get("semantic_mask")
        if semantic_mask is not None:
            out["semantic_mask"] = comfy.conds.CONDRegular(semantic_mask)
        source_latents = kwargs.get("source_latents")
        if source_latents is not None:
            out["source_latents"] = comfy.conds.CONDRegular(self.process_latent_in(source_latents))
        return out
