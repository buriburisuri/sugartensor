# -*- coding: utf-8 -*-
import collections
# import Queue
# import inspect
# import sg_logging
# import os
# import threading
# import warnings
#
# import joblib
# from tqdm import tqdm
#
# from sg_data import BaseDataSet

__author__ = 'njkim@jamonglab.com'


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


#
# class FileUtil(object):
#     @staticmethod
#     def save(obj, path, compress=True):
#
#         import cPickle
#         import joblib
#
#         # make dir if not exist
#         if not os.path.exists(os.path.split(path)[0]):
#             os.makedirs(os.path.split(path)[0])
#
#         if compress:
#             joblib.dump(obj, path, compress=3, cache_size=100, protocol=cPickle.HIGHEST_PROTOCOL)
#         else:
#             joblib.dump(obj, path)
#
#     @staticmethod
#     def load(path):
#         try:
#             warnings.filterwarnings('ignore')
#             return joblib.load(path, mmap_mode='r+')
#         except IOError:
#             return None
#
#
# class ThreadUtil(object):
#     @staticmethod
#     def threaded(f):
#
#         def _main_thread_check():
#             for t in threading.enumerate():
#                 if t.name == 'MainThread':
#                     return t.is_alive()
#
#         def _wrapped(q, *args, **kwargs):
#             """
#             this function calls the decorated function and puts the result in a queue
#             """
#             if 'mtc' in inspect.getargspec(f).args:
#                 q.put(f(mtc=_main_thread_check, *args, **kwargs))
#             else:
#                 q.put(f(*args, **kwargs))
#
#         def _wrap(*args, **kwargs):
#             """
#             this is the function returned from the decorator. It fires off
#             wrapped_f in a new thread and returns the thread object with
#             the result queue attached
#             """
#             q = Queue.Queue()
#             t = threading.Thread(target=_wrapped, args=(q,) + args, kwargs=kwargs)
#             t.start()
#             t.result = q
#             return t
#
#         return _wrap
#
#     @staticmethod
#     class InterruptableQueue(Queue.Queue):
#         """
#         keyboard interruptable queue
#         """
#
#         def i_put(self, obj):
#             while ThreadUtil._is_main_alive():
#                 try:
#                     self.put_nowait(obj)
#                     break
#                 except Queue.Full:
#                     pass
#
#         def i_get(self):
#             while ThreadUtil._is_main_alive():
#                 try:
#                     return self.get_nowait()
#                 except Queue.Empty:
#                     pass
#
#     @staticmethod
#     def _is_main_alive():
#         """
#         check if main thread is alive
#         """
#         return any((i.name == "MainThread") and i.is_alive() for i in threading.enumerate())
#
#
# class ProgressBar(tqdm):
#     def __init__(self, iterable=None, **kwargs):
#         kwargs = Opt(kwargs)
#         # if feeder provided
#         if isinstance(iterable, BaseDataSet):
#             kwargs += Opt(set_name='train', column=[])
#             total = iterable.total_batch(kwargs.set_name)
#             desc = kwargs.set_name
#             iterable = iterable._next_data(kwargs.set_name, kwargs.column)
#             del kwargs['set_name']
#             del kwargs['column']
#         else:
#             total, desc = None, None
#
#         # call super
#         super(self.__class__, self).__init__(iterable=iterable, total=total, desc=desc,
#                                              ncols=80, unit='b', leave=False, **kwargs)


#
# aliases
#

# option
sg_opt = _Opt
#
# # logger
# _logger = Logger()
# debug = _logger.debug
# warn = _logger.warn
# info = _logger.info
# error = _logger.error
# critical = _logger.critical
#
# # file util
# load = FileUtil.load
# save = FileUtil.save
#
#
# # thread util
# threaded = ThreadUtil.threaded
# InterruptableQueue = ThreadUtil.InterruptableQueue
#
#
# # probrass bar
# pbar = ProgressBar
