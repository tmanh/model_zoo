import mmcv
import torch
import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from mmengine.model import BaseModule


from ..universal.depthformer_utils import colorize


class BaseDepther(BaseModule, metaclass=ABCMeta):
    """Base class for depther."""

    def __init__(self, init_cfg=None):
        super(BaseDepther, self).__init__(init_cfg)

    @abstractmethod
    def encode(self, imgs):
        """Placeholder for extract features from images."""
        pass

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def show_result(self, img, result, win_name='', show=False, wait_time=0, out_file=None, format_only=False):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The depth estimation results.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param. Default: 0.
            show (bool): Whether to show the image. Default: False.
            out_file (str or None): The filename to write the image. Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        depth = result[0]

        if show:
            mmcv.imshow(img, win_name, wait_time)

        if out_file is not None:
            if format_only:
                np.save(out_file, depth) # only save the value.
            else:
                depth = colorize(depth, vmin=self.decode_head.min_depth, vmax=self.decode_head.max_depth)
                mmcv.imwrite(depth.squeeze(), out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only result depth will be returned')
            return depth
