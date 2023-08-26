from mmengine.model import BaseModule
from mmengine.config import Config

from ..universal.depthformer_basics import DEPTHER, build_depther, build_head, build_neck, build_backbone


def build_depther_from(path):
    cfg = Config.fromfile(path)
    return build_depther(cfg.model)


@DEPTHER.register_module()
class DepthEncoderDecoder(BaseModule):
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_decode_head(self):
        return hasattr(self, 'decode_head') and self.decode_head is not None

    def __init__(self, backbone, decode_head, neck=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(DepthEncoderDecoder, self).__init__(init_cfg)
        # TODO: test pretrained

        if pretrained is not None:
            assert backbone.get('pretrained') is None, 'both backbone and depther set pretrained weight'
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.init_decoder(decode_head)

        self.list_feats = self.decode_head.in_channels[::-1]
        self.level2full = self.decode_head.level2full

        if neck is not None:
            self.neck = build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def init_decoder(self, decode_head):
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners

    def encode(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_run(self, img):
        x = self.encode(img)
        return self.decode_head(x)

    def extract_feats(self, img):
        x = self.encode(img)
        return self.decode_head(x), x

    def set_depth_range(self, min_depth, max_depth):
        self.backbone.min_depth = min_depth
        self.backbone.max_depth = max_depth
