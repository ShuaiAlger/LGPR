import torch
import torch.nn as nn
import torchvision.transforms
import abc




class Backbone2Base(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abc.abstractmethod
    def get_feature(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_alignment(self):
        raise NotImplementedError

    def forward(self, x):
        return self.get_feature(x)
    
MODEL_INFO = {
    'dinov2_vits14': {'n_channels': 384, 'alignment': 14},
    'dinov2_vitb14': {'n_channels': 768, 'alignment': 14},
    'dinov2_vitl14': {'n_channels': 1024, 'alignment': 14},
    'dinov2_vitg14': {'n_channels': 1536, 'alignment': 14},
    'dinov2_vits14_reg': {'n_channels': 384, 'alignment': 14},
    'dinov2_vitb14_reg': {'n_channels': 768, 'alignment': 14},
    'dinov2_vitl14_reg': {'n_channels': 1024, 'alignment': 14},
    'dinov2_vitg14_reg': {'n_channels': 1536, 'alignment': 14},
}


class DINOv2(Backbone2Base):
    def __init__(self, model_name: str, pretrained=True, feature_type='patch'):
        super().__init__()

        assert model_name in MODEL_INFO
        assert feature_type in ['patch', 'global']
        # 'patch': patch tokens, 'global': cls concat w/ avgpooled patch tokens

        self.model_name = model_name
        self.feature_type = feature_type

        # Load Model
        self.model = None
        self.preprocessor = None

        self._load_model(model_name, pretrained)
        self._load_preprocessor()

        info = MODEL_INFO[model_name]

        if self.feature_type == 'patch':
            self._num_channels = info['n_channels']
        else:
            self._num_channels = info['n_channels'] * 2

        self._alignment = info['alignment']

    def _load_model(self, model_name: str, pretrained=True):
        self.model = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=pretrained)

    def _load_preprocessor(self):
        self.preprocessor = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

    def get_num_channels(self):
        return self._num_channels

    def get_alignment(self):
        return self._alignment

    def get_feature(self, x):
        x = self.preprocessor(x)

        if self.feature_type == 'patch':
            bs, c, h, w = x.shape
            ret = self.model.forward_features(x)['x_norm_patchtokens']
            ret = ret.reshape(bs, h // self._alignment, w // self._alignment, self._num_channels)
        else:
            features = self.model.forward_features(x)
            ret = torch.cat(
                (features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(dim=1)), dim=1)

        return ret



class Condense(nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    def __init__(
        self,
        backbone_name="dinov2_vitg14",
        num_unfrozen_blocks=0,
        return_cls_token=False
        ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.return_cls_token = return_cls_token

        weights_file = "/media/shuai/Correspondence/VPR/condense/ckpt/condense_vitg14.pth"

        self.backbone = DINOv2(backbone_name, pretrained=False, feature_type='patch')
        try:
            self.dino.model.load_state_dict(
                torch.load(weights_file), strict=True)
        except AttributeError:
            pass

        # self.backbone.requires_grad_(False)
        # self.backbone.model.requires_grad_(False)
        for blk in self.backbone.model.blocks[ : -self.num_unfrozen_blocks]:
            blk.requires_grad_(False)

        self.out_channels = 1536

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.backbone.model.prepare_tokens_with_masks(x)
            for blk in self.backbone.model.blocks[ : -self.num_unfrozen_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.backbone.model.blocks[-self.num_unfrozen_blocks : ]:
            x = blk(x)
            
        x_cls = x[:, 0]
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        _, _, C = x.shape # we know C == self.dino.embed_dim, but still...
        x = x.permute(0, 2, 1).contiguous().view(B, C, H//14, W//14)
        
        if self.return_cls_token:
            return x, x_cls
        
        return x
