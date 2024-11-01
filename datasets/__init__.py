from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .stl import STL10
from .utils import *
from .randaugment import RandAugment
from .covid19 import COVID19
__all__ = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10','COVID19')
