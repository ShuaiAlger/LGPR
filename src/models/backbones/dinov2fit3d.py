import torch
import torch.nn as nn

import types

def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)

class DinoV2Fit3D(nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=2,
        return_cls_token=False,
    ):
        """DinoV2 backbone with the ability to keep only the last num_unfrozen_blocks trainable.

        Args:
            backbone_name (str, optional): DinoV2 variant. Defaults to "dinov2_vitb14".
            num_unfrozen_blocks (int, optional): number of blocks to unfreeze. Defaults to 2.

        Raises:
            ValueError: if the backbone_name is not in the available models.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.return_cls_token = return_cls_token
        # make sure the backbone_name is in the available models
        # if self.backbone_name not in self.AVAILABLE_MODELS:
        #     raise ValueError(f"Backbone {self.backbone_name} is not recognized!" 
        #                      f"Supported backbones are: {self.AVAILABLE_MODELS}")


        self.fine_models = {}        
        self.fine_models["DINOv2"] = torch.hub.load("ywyue/FiT3D", 'dinov2_small_fine').to("cuda")

        self.fine_models["DINOv2"].get_intermediate_layers = types.MethodType(
                                        get_intermediate_layers,
                                        self.fine_models["DINOv2"]
                                    )


        
        self.out_channels = 384

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers

        fine_feats = self.fine_models["DINOv2"].get_intermediate_layers(x, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                            return_class_token=False, norm=True)

        feat = fine_feats[-1]
        
        return feat