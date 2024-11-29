import os

from .small_xception import SmallXception

os.environ['KERAS_BACKEND'] = 'tensorflow'

__all__ = ['SmallXception']
