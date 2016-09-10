# -*- coding: utf-8 -*-
import collections

__author__ = 'buriburisuri@gmail.com'


class _Opt(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

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
        res = _Opt(self.__dict__)
        for k, v in other.iteritems():
            if k not in res.__dict__ or res.__dict__[k] is None:
                res.__dict__[k] = v
        return res

    def __mul__(self, other):
        res = _Opt(self.__dict__)
        for k, v in other.iteritems():
            res.__dict__[k] = v
        return res


# aliasing
sg_opt = _Opt
