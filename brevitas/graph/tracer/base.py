import functools
from abc import abstractmethod

from ..module import FnType


# Adapted from: https://bit.ly/3hYCpvJ (stackoverflow)
class TracerMeta(type):

    # Adapted from: https://code.activestate.com/recipes/496741-object-proxying/
    magic_methods = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', '__contains__',
        '__delitem__', '__delslice__', '__div__', '__divmod__', '__eq__', '__float__',
        '__floordiv__', '__ge__', '__getitem__', '__getslice__', '__gt__', '__hash__',
        '__hex__', '__iadd__', '__iand__', '__idiv__', '__idivmod__', '__ifloordiv__',
        '__ilshift__', '__imod__', '__imul__', '__int__', '__invert__', '__ior__', '__ipow__',
        '__irshift__', '__isub__', '__itruediv__', '__ixor__', '__le__', '__len__',
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__oct__',
        '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdiv__', '__rdivmod__',
        '__reduce__', '__reduce_ex__', '__reversed__', '__rfloorfiv__',  '__rlshift__', '__rmod__',
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__',
        '__rxor__', '__setitem__', '__setslice__', '__sub__', '__truediv__', '__xor__',
        '__next__', '__iter__']

    @staticmethod
    def _magic_function(tracer, method_name, *args, **kwargs):
        if hasattr(tracer.value_, method_name) and method_name in TracerMeta.magic_methods:
            fn = getattr(tracer.value_, method_name)
            args, kwargs = tracer.repack_args_kwargs(args, kwargs)
            out = fn(*args, **kwargs)
            kwargs['self'] = tracer.value_
            inplace = tracer.is_inplace_function(out, args, kwargs)
            out = tracer.update_inplace_output(inplace, out)
            out = tracer.update_trace(method_name, FnType.METHOD, args, kwargs, out)
            return tracer.epilogue(inplace, out)

    def __new__(cls, name, bases, attr):
        new = super(TracerMeta, cls).__new__(cls, name, bases, attr)
        for method_name in TracerMeta.magic_methods:
            magic_method = functools.partialmethod(TracerMeta._magic_function, method_name)
            setattr(new, method_name, magic_method)
        return new


class TracerBase(metaclass=TracerMeta):

    @abstractmethod
    def _trace_oop_function(self, original_fn, args, kwargs):
        pass
