# Copyright (c) 2020 Lightricks. All rights reserved.

import logging
from os.path import split, isdir
from os import mkdir
import sys
from cnvrg import Experiment

try:
    e = Experiment()
except:
    e = None

def is_experiment():
    try:
        e = Experiment()
        e['title']
    except:
        return False
    return True

def cnvrg_print(key, val):
    print('cnvrg_tag_%s: %s'%(key, str(val)))

def cnvrg_tag(key, val):
    e.log_param(key, val)


def cvnrg_linechart(chart_name, key, value, group=None):
    if is_experiment():
        e.log_metric(chart_name,
                    Ys=[value],
                    Xs=[key],
                    grouping=[group])
    else:
       if group is None:
           print("cnvrg_linechart_{} key: '{}' value: '{}'".format(chart_name.replace(' ','_'), key, value))
       else:
           print("cnvrg_linechart_{} group: '{}' key: '{}' value: '{}'\n".format(chart_name.replace(' ','_'), group, key, value))


def print_args(args):
    for k in args.__dict__:
        v = args.__dict__[k]
        cnvrg_print(k, v)


import numpy as np
import time
def tic():
    global start_time
    start_time = time.time()


def toc():
    t = time.time()-start_time
    if t < 60:
        print(t,'sec')
    else:
        print(t/60,'min')


def toc2():
    return (time.time()-start_time)


def measure_inference_time(model, x):
    for _ in range(50):
        model.predict(np.array([x]))
    times = []
    for _ in range(100):
        tic()
        model.predict(np.array([x]))
        times.append(toc2())
    return np.mean(times)


def print_experiment():
    try:
        e = Experiment()
        print('title:', e['title'])
        print('URL:', e['full_href'])
    except:
        pass


def redirect_outputs(log_path):
    check_path(split(log_path)[0])
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    fh.terminator = sh.terminator = ""

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    class LogStream(object):
        def __init__(self, log_level):
            self._log_level = log_level

        def write(self, *args, **kwargs):
            if self._log_level.lower() == 'debug':
                logger.debug(*args, **kwargs)
            elif self._log_level.lower() == 'info':
                logger.info(*args, **kwargs)
            elif self._log_level.lower() == 'warn':
                logger.warning(*args, **kwargs)
            elif self._log_level.lower() == 'error':
                logger.error(*args, **kwargs)
            else:
                pass

        def flush(self, *args, **kwargs):
            pass

    sys.stdout = LogStream('info')
    sys.stderr = LogStream('error')

def check_path(path):
    if not isdir(path):
        check_path(split(path)[0])
        mkdir(path)
