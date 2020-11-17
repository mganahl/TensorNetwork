import numpy as np
import time

timings={}

def block_until_ready(args):
  if hasattr(args, "block_until_ready"):
    args.block_until_ready()

  else:
    if hasattr(args, '__len__'):
      if len(args) > 0:
        if hasattr(args[0], "block_until_ready"):
          args[0].block_until_ready()


def timer(fun, name, *args, **kwargs):
  def timed_fun(*args, **kwargs):
    t1 = time.time()
    result = fun(*args, **kwargs)
    block_until_ready(result)
    t2 = time.time()
    ts = timings.get(name, [])
    ts.append(t2 - t1)
    timings[name] = ts
    return result
  return timed_fun

def reset_all():
  global timings
  timings = {}

def reset(name):
  global timings
  timings[name] = []
