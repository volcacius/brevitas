import builtins
import math
import inspect

from .base import TracerBase

math_module_functions = [
    *inspect.getmembers(math, inspect.isbuiltin)]

math_builtins_functions = [
    ('max', builtins.max),
    ('min', builtins.min),
    #    ('sum', builtins.sum),
]


def math_function_wrapper(original_fn, args, kwargs):
    tracer = None
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, (list, tuple)):
            for a in arg:
                if isinstance(a, TracerBase):
                    tracer = a
                    break
        elif isinstance(arg, TracerBase):
            tracer = arg
            break  # TODO: we should timestamp the tracer and use the latest one instead
    if tracer is not None:
        return tracer._trace_oop_function(original_fn, args, kwargs)
    else:
        return original_fn(*args, **kwargs)