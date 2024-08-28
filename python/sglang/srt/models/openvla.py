import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import timm
import timm.data
import tokenizers
import torch
import torch.nn as nn
import transformers
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.openvla_layers import PrismaticProjector, PrismaticVisionBackbone
from sglang.srt.model_executor.forward_batch_info import InputMetadata
from sglang.srt.models.llama2 import LlamaForCausalLM

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.model": "model",
    "language_model.lm_head": "lm_head",
}

TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"],
    "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None],
    "in1k-vit-l": [None],
    "siglip-vit-so400m": [None],
    "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None],
    "dinosiglip-vit-so-384px": [None, None],
}

LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama",
}

logger = logging.getLogger(__name__)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    next_token_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class OpenVLAConfig(PretrainedConfig):
    model_type: str = "openvla"
    is_composition: bool = False

    def __init__(
        self,
        norm_stats: Optional[
            Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
        ] = None,
        n_action_bins: int = 256,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(
                self.vision_backbone_id.startswith(v)
                for v in ["dinoclip", "dinosiglip"]
            )
        )

        self.timm_model_ids = [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = [224, 224]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = "meta-llama/Llama-2-7b-hf"
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](
                **text_config
            )
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.vocab_size = 32064
        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAForActionPrediction(PreTrainedModel):
    config_class: PretrainedConfig = OpenVLAConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def __init__(
        self,
        config: OpenVLAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__(config)
        self.embeddings_layer = None
        self.past_key_values = None
        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )
        if (transformers.__version__ != "4.40.1") or (
            tokenizers.__version__ != "0.19.1"
        ):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config=quant_config
        )
        # self.language_model = AutoModelForCausalLM.from_config(
        #     config.text_config, attn_implementation=config._attn_implementation
        # )

        # self.processor = PrismaticProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = (
            self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        )

    def pad_input_ids(
        self, input_ids, pad_value, pt_shape=None, image_size=None, multiple_of=64
    ):
        image_pad_len = ((pt_shape[-1] - 1) // multiple_of + 1) * multiple_of
        input_ids = input_ids[:1] + [pad_value] * image_pad_len + input_ids[1:]
        if input_ids[-1] != 29871:
            input_ids.append(29871)

        return input_ids, 1

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)
        new_weights = []
        params_dict = dict(self.named_parameters())
        for name, weight in weights:
            if not "language_model" in name:
                param = params_dict[name]
                default_weight_loader(param, weight)
                continue

            new_name = None
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    new_name = name.replace(key_to_modify, new_key)

            if new_name is not None:
                new_weights.append((new_name, weight))
            else:
                new_weights.append((name, weight))

        weights = new_weights

        self.language_model.load_weights(weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        pixel_values: Optional[List[Optional[np.array]]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_offsets: Optional[List[int]] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        need_vision = pixel_values is not None and any(
            p is not None for p in pixel_values
        )

        # === Handle Unimodal Forward ===
        if not need_vision:
            assert (
                input_ids is not None
            ), "Missing `input_ids` in language-only forward!"
            return self.language_model(
                input_ids=input_ids,
                positions=positions,
                input_metadata=input_metadata,
                input_embeds=None,
            )

        # === Handle Multimodal Forward ===
        # embedding_layer = self.language_model.model.embed_tokens 
        unpadded_input_ids = input_ids[input_ids != -1].unsqueeze(0)
        embedding_layer = self.get_embedding_layer_from_file()
        input_embeddings = embedding_layer(unpadded_input_ids)
        assert(len(pixel_values) ==1, "OpenVLA only supports one pixel values as input")

        patch_features = self.vision_backbone(pixel_values[0])
        projected_patch_embeddings = self.projector(patch_features)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_embeddings = multimodal_embeddings.squeeze(0)
        # print("multimodal_embeddings", multimodal_embeddings)
        return self.language_model(
            input_ids=None,
            positions=positions,
            input_metadata=input_metadata,
            input_embeds=multimodal_embeddings,
        )
    
    def get_embedding_layer_from_file(self) -> nn.Module:
        if self.embeddings_layer == None:
            from huggingface_hub import hf_hub_download
            self.embeddings_layer = torch.load(hf_hub_download(repo_id="depetrol/openvla-7b", filename="embedding_layer.pt"), weights_only=False).to('cuda')
        return self.embeddings_layer

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """Thin wrapper around super().generate() that decodes predicted actions and de-normalizes them."""

        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(
                        input_ids.device
                    ),
                ),
                dim=1,
            )

        # Run VLA inference
        generated_ids = self.generate(
            input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs
        )
        # generated_ids = self.generate(input_ids, max_new_tokens=7, **kwargs)

        # Run VLA inference without KV-Cache
        # generated_ids = input_ids.clone()
        # self.past_key_values = None
        # for _ in range(7):
        #     output = self.generate(generated_ids, max_new_tokens=1, **kwargs)
        #     generated_ids = torch.cat((generated_ids, output[:, -1:]), dim=-1)

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = (
            generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        )
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(
        norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]
    ) -> str:
        if unnorm_key is None and len(norm_stats) != 1:
            raise ValueError(
                f"Your model was trained on more than one dataset. "
                f"Please pass a `unnorm_key` from the following options to choose the statistics used for "
                f"de-normalizing actions: {norm_stats.keys()}"
            )

        # If None, grab the (singular) dataset in `norm_stats` to use as `unnorm_key`
        unnorm_key = (
            unnorm_key if unnorm_key is not None else next(iter(norm_stats.keys()))
        )
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {norm_stats.keys()}"
            )

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def tie_weights(self) -> None:
        return
        # self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)


EntryClass = OpenVLAForActionPrediction
