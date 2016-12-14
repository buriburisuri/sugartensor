from __future__ import absolute_import
import six
import collections


__author__ = 'buriburisuri@gmail.com'


class _Opt(collections.MutableMapping):
    r"""Option utility class.

    This class is only internally used for sg_opt.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    # noinspection PyUnusedLocal,PyUnusedLocal
    def __getattr__(self, key):
        return None

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __add__(self, other):
        r"""Overloads `+` operator.
        
        It does NOT overwrite the existing item.
        
        For example,
        
        ```python
        import sugartensor as tf

        opt = tf.sg_opt(size=1)
        opt += tf.sg_opt(size=2)
        print(opt) # Should be {'size': 1}
        ```
        """
        res = _Opt(self.__dict__)
        for k, v in six.iteritems(other):
            if k not in res.__dict__ or res.__dict__[k] is None:
                res.__dict__[k] = v
        return res

    def __mul__(self, other):
        r"""Overloads `*` operator.
        
        It overwrites the existing item.
        
        For example,
        
        ```python
        import sugartensor as tf

        opt = tf.sg_opt(size=1)
        opt *= tf.sg_opt(size=2)
        print(opt) # Should be {'size': 2}
        ```
        """
        res = _Opt(self.__dict__)
        for k, v in six.iteritems(other):
            res.__dict__[k] = v
        return res


# aliasing
sg_opt = _Opt
