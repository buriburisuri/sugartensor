# -*- coding: utf-8 -*-
import sugartensor as tf
from functools import wraps
import threading
from tensorflow.python.platform import tf_logging as logging


__author__ = 'buriburisuri@gmail.com'


def sg_producer_func(func):
    @wraps(func)
    def wrapper(**kwargs):

        # default option
        opt = tf.sg_opt(kwargs) + tf.sg_opt(dtypes=[tf.sg_floatx], capacity=32, num_threads=1)

        # source queue list check
        assert opt.source is not None, 'source is mandatory.'
        if type(opt.source) is not list and type(opt.source) is not tuple:
            opt.source = [opt.source]
        if type(opt.dtypes) is not list and type(opt.dtypes) is not tuple:
            opt.dtypes = [opt.dtypes]
        assert len(opt.source) == len(opt.dtypes), 'Source and dtypes shoud have same length.'

        # enqueue function
        def enqueue_func(sess, op, coord):
            # read data from source queue
            data = func(sess.run(opt.source))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in opt.dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(opt.capacity, dtypes=opt.dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = FuncQueueRunner(enqueue_func, queue, [enqueue_op] * opt.num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                              queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op, coord)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs -= 1
                        decremented = True
                        if self._runs == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs -= 1






