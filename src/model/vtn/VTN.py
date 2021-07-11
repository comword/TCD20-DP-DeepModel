import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from base.base_model import BaseModel
from .options import VTNOptions
from .vtn_helper import VTNLongformerModel, pad_to_window_size_local


# https://github.com/bomri/SlowFast/blob/master/slowfast/models/video_model_builder.py


class VTN(BaseModel):
    """
    VTN model builder.
    Daniel Neimark, Omri Bar, Maya Zohar and Dotan Asselmann.
    "Video Transformer Network."
    https://arxiv.org/abs/2102.00719
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VTN, self).__init__()
        self._construct_network(cfg)

    def _construct_network(self, cfg: VTNOptions):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        if cfg.backbone.startswith("efficientnet"):
            self.backbone = EfficientNet.from_pretrained(
                cfg.backbone, include_top=False, num_classes=0)
            self.backbone.set_swish(memory_efficient=False)
        else:
            raise NotImplementedError(f"not supporting {cfg.backbone}")

        x = torch.zeros((1,) + cfg.input_shape)
        infer_shape = self.backbone(x).shape

        embed_dim = infer_shape[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.temporal_encoder = VTNLongformerModel(
            embed_dim=embed_dim,
            max_position_embeddings=cfg.MAX_POSITION_EMBEDDINGS,
            num_attention_heads=cfg.NUM_ATTENTION_HEADS,
            num_hidden_layers=cfg.NUM_HIDDEN_LAYERS,
            attention_mode=cfg.ATTENTION_MODE,
            pad_token_id=cfg.PAD_TOKEN_ID,
            attention_window=cfg.ATTENTION_WINDOW,
            intermediate_size=cfg.INTERMEDIATE_SIZE,
            attention_probs_dropout_prob=cfg.ATTENTION_PROBS_DROPOUT_PROB,
            hidden_dropout_prob=cfg.HIDDEN_DROPOUT_PROB)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.Linear(cfg.HIDDEN_DIM, cfg.MLP_DIM),
            nn.GELU(),
            nn.Dropout(cfg.MLP_DROPOUT_RATE),
            nn.Linear(cfg.MLP_DIM, cfg.n_classes)
        )

    def forward(self, x):

        x, position_ids = x

        # spatial backbone
        B, C, F, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)
        x = x.reshape(B, F, -1)

        # temporal encoder (Longformer)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = torch.zeros(
            x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        # # MLP head
        # x = x["last_hidden_state"]
        # x = self.mlp_head(x[:, 0])
        return x
