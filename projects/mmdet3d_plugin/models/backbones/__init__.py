from mmdet.models.backbones import ResNet
from .resnet import CustomResNet,MyCustomResNet
from .swin import SwinTransformer
# from .sparse_resnet import SparseResnet3D
from .ghh_backbone import GHH_Backbone

# __all__ = ['ResNet', 'CustomResNet', 'SwinTransformer', 'MyCustomResNet', 'SparseResnet3D']
# __all__ = ['ResNet', 'CustomResNet', 'SwinTransformer', 'MyCustomResNet']
__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer', 'MyCustomResNet', 'GHH_Backbone']

