import logging
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import CudaGraphManager, is_graph_capturing
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer, get_transformer_layer_offset
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


@dataclass
class ShortcutTransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a shortcut connected transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention_1: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda_1: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_1_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp_1: Union[ModuleSpec, type] = IdentityOp
    mlp_bda_1: Union[ModuleSpec, type] = IdentityFuncOp

    self_attention_2: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda_2: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_2_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp_2: Union[ModuleSpec, type] = IdentityOp
    mlp_bda_2: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_3_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp_3: Union[ModuleSpec, type] = IdentityOp
    mlp_bda_3: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class ShortcutTransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single shortcut connected transformer layer.

    Shortcut connected Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: ShortcutTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        # Enable cuda graphs.
        if (
            config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration"
        ) or config.external_cuda_graph:
            assert not (
                config.enable_cuda_graph and config.external_cuda_graph
            ), "Cudagraphs and external cudagraphs cannot be enabled at the same time"
            if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
                if not self.training:
                    # Cudagraphs for inference are only enabled with the flash decoding kernel
                    assert (
                        self.config.flash_decode
                    ), "--flash-decode is required to use CUDA graphs during inference"
                self.cudagraph_manager = CudaGraphManager(config, vp_stage=vp_stage)
            else:
                # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
                # the number of microbatches. Multiple CUDA graphs per layer is required to support
                # pipelining which requires running FWD graph of multiple microbatches before BWD
                # graph. To enable CUDA graph, this list should be populated in the model training
                # script with the graphs returned by make_graphed_callables API before the first
                # training step.
                self.cuda_graphs = []
                # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
                # graphs. Those hooks and args are collected in this list and should be manually
                # triggered before CUDA Graph running. This is required to ensure the correct param
                # all-gather overlap with forward compute.
                self.cuda_graph_manual_hooks = []
                self.current_microbatch = -1

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Module 2: SelfAttention-1]
        self.self_attention_1 = build_module(
            submodules.self_attention_1,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion-1]
        self.self_attn_bda_1 = build_module(submodules.self_attn_bda_1)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_1_layernorm = build_module(
            submodules.pre_mlp_1_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        additional_mlp_kwargs = {}
        # import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp_1, ModuleSpec):
            if submodules.mlp_1.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["pg_collection"] = pg_collection
            elif submodules.mlp_1.module == MLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp_1.module == TEFusedMLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp_1)}. Using default kwargs.",
                )
        self.mlp_1 = build_module(submodules.mlp_1, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp_1, 'set_layer_number'):
            self.mlp_1.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda_1 = build_module(submodules.mlp_bda_1)

        # [Module 10: SelfAttention-2]
        self.self_attention_2 = build_module(
            submodules.self_attention_2,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 11: BiasDropoutFusion-2]
        self.self_attn_bda_2 = build_module(submodules.self_attn_bda_2)

        # [Module 12: Pre MLP-2] Optional Layernorm before MLP-2
        self.pre_mlp_2_layernorm = build_module(
            submodules.pre_mlp_2_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mlp_2 = build_module(submodules.mlp_2, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp_2, 'set_layer_number'):
            self.mlp_2.set_layer_number(self.layer_number)

        # [Module 13: BiasDropoutFusion]
        self.mlp_bda_2 = build_module(submodules.mlp_bda_2)

        # [Module 13: Pre MLP-3] Optional Layernorm before MLP-3
        self.pre_mlp_3_layernorm = build_module(
            submodules.pre_mlp_3_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mlp_3 = build_module(submodules.mlp_3, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp_3, 'set_layer_number'):
            self.mlp_3.set_layer_number(self.layer_number)

        # [Module 13: BiasDropoutFusion]
        self.mlp_bda_3 = build_module(submodules.mlp_bda_3)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if (
                    not isinstance(self.input_layernorm, IdentityOp)
                    and not self.config.external_cuda_graph
                ):
                    self.recompute_input_layernorm = True
                    if self.config.fp8:
                        self.self_attention_1.set_for_recompute_input_layernorm()
                        self.self_attention_2.set_for_recompute_input_layernorm()
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8:
                        if isinstance(self.mlp_1, MoELayer):
                            self.mlp_1.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import (
                                set_save_original_input,
                            )

                            set_save_original_input(self.mlp_1.linear_fc1)
            if "mlp" in self.config.recompute_modules:
                if not isinstance(self.mlp_1, MoELayer):
                    self.recompute_mlp = True

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "ShortcutTransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)
    
    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the shortcut transformer layer.

        This method calls the core computation of a shortcut transformer layer, including
        self-attention-1, cross-attention-1 (if applicable), and feed-forward operations,
        self-attention-2, cross-attention-1 (if applicable), and feed-forward operations,
        and shortcut feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
        hidden_states, context = self._forward_attention_1(*args, **kwargs)
        output_1 = self._forward_mlp_3(hidden_states, kwargs.get("inference_context", None))
        hidden_states = self._forward_mlp_1(hidden_states, kwargs.get("inference_context", None))
        hidden_states, context = self._forward_attention_2(*args, **kwargs)
        output_2 = self._forward_mlp_2(hidden_states, kwargs.get("inference_context", None))
        output = output_1 + output_2

        return output, context
