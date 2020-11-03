from contextlib import ExitStack
from _collections_abc import Iterable
from inspect import getcallargs
from typing import List, Union, Any
from copy import copy
from packaging import version


import torch
from torch import Tensor
from torch.nn import Module, Sequential, ModuleList, ModuleDict

from brevitas.quant_tensor import QuantTensor
from .base import TracerBase
from .trace import Trace, TraceElem
from .wrapper.scriptmodule import torchscript_wrapper
from .wrapper.builtin import IntWrapper, StrWrapper, FloatWrapper
from ..module import FnType
from ..utils import flatten, module_class_name

if version.parse(torch.__version__) > version.parse('1.6.0'):
    from torch.overrides import get_testing_overrides
    from .patch import ABOVE_16_PATCHES as PATCHES
elif version.parse(torch.__version__) == version.parse('1.6.0'):
    from torch._overrides import get_testing_overrides
    from .patch import EQUAL_16_PATCHES as PATCHES
else:
    from .backport.signatures import get_testing_overrides
    from .patch import BELOW_16_PATCHES as PATCHES

TORCH_FN_NAMES = [fn.__name__ for fn in get_testing_overrides().keys()]
TORCH_FN_OVERRIDE_DICT = get_testing_overrides()

TYPES_TO_TRACE = [
    IntWrapper,
    FloatWrapper,
    StrWrapper,
    Iterable,  # works with tuple_iterator and such
    list,
    dict,
    tuple,
    torch.Tensor,
    QuantTensor,
    torch.Size]


class CallableWrapper(object):

    def __init__(self, tracer, callabl, inplace):
        self.tracer = tracer
        self.callabl = callabl
        self.inplace = inplace

    def __call__(self, *args, **kwargs):
        args, kwargs = self.tracer.repack_args_kwargs(args, kwargs)
        out = self.callabl(*args, **kwargs)
        kwargs['self'] = self.tracer.value_
        out = self.tracer.update_trace(self.callabl.__name__, FnType.METHOD, args, kwargs, out)
        return self.tracer.epilogue(self.inplace, out)


class Tracer(TracerBase):

    def __init__(self, value_, trace_ = None, namespace_= None):
        self.value_ = value_
        self.trace_ = trace_ or Trace()
        self.namespace_ = namespace_ or []
        self.hook_handles = []
        self.preserve_module_allowlist: List = ['torch.nn', 'brevitas.nn']
        self.preserve_module_blocklist: List = [Sequential, ModuleList, ModuleDict]

    def __bool__(self):
        return self.value_

    def __getattr__(self, name: str):
        inplace_tensor = name.endswith('_') and name[:-1] in TORCH_FN_NAMES
        attr = getattr(self.value_, name)
        if callable(attr):
            return CallableWrapper(self, attr, inplace_tensor)
        else:
            kwargs = {'self': self.value_}
            self.update_trace(name, FnType.ATTRIBUTE, [], kwargs, attr)
            return self.epilogue(False, attr)

    def is_inplace_function(self, out, fn_args, fn_kwargs):
        return any([out is arg for arg in fn_args + list(fn_kwargs.values())])

    # TODO merge with tensor function
    def _trace_oop_function(self, function, args, kwargs):
        args, kwargs = self.repack_args_kwargs(args, kwargs)
        out = function(*args, **kwargs)
        out = self.update_trace(function, FnType.FUNCTION, args, kwargs, out)
        return self.epilogue(False, out)

    def trace_module_through(self, module):
        return (isinstance(module, tuple(self.preserve_module_blocklist))
                or all([not module_class_name(module).startswith(pma)
                        for pma in self.preserve_module_allowlist]))

    def is_tracing(self, input_list):
        return any(isinstance(i, Tracer) for i in flatten(input_list))

    def repack_value(self, value, check_already_in_map=True):
        value_from_map = self.trace_.index_from_map(value)
        if not check_already_in_map or value_from_map is None:
            if isinstance(value, Tracer):
                return value.value_
            elif isinstance(value, int) and not isinstance(value, bool):
                return IntWrapper(value)
            elif isinstance(value, float):
                return FloatWrapper(value)
            elif isinstance(value, str):
                return StrWrapper(value)
            elif isinstance(value, tuple) and not isinstance(value, QuantTensor) and not isinstance(value, torch.Size):
                return tuple(self.repack_value(v) for v in value)
            elif isinstance(value, list):
                return [self.repack_value(v) for v in value]
            elif isinstance(value, dict):
                return {self.repack_value(k): self.repack_value(v) for k, v in value.items()}
            else:
                return value
        else:
            return value

    def repack_args_kwargs(self, args, kwargs):
        new_args = [self.repack_value(a) for a in args]
        new_kwargs = {k: self.repack_value(a) for k, a in kwargs.items()}
        return new_args, new_kwargs

    def repack_module_input(self, value):
        if isinstance(value, list) or (isinstance(value, tuple) and len(value) == 1):
            value = [self.repack_value(v) for v in value]
        else:
            value = [self.repack_value(value)]
        return value

    def repack_model_output(self, value):
        if isinstance(value, (list, tuple)):
            value = self.repack_value(value)
        else:
            value = [self.repack_value(value)]
        return value

    def epilogue(self, inplace, output):
        if inplace:
            assert not isinstance(output, tuple)  # sanity check
            self.value_ = output
            return self
        elif isinstance(output, tuple(TYPES_TO_TRACE)):
            new_tracer = Tracer(trace_=self.trace_, namespace_=self.namespace_, value_=output)
            return new_tracer
        else:
            return output

    def _register_forward_pre_hooks(self, model: torch.nn.Module):

        def enter_namespace(module, input):
            name = None
            if not self.namespace_:  # root module
                name = ''
            else:
                supermodule = self.namespace_[-1][0]
                for name, mod in supermodule._modules.items():
                    # identify current module as a submodule
                    if mod is module:
                        break
            input_list = self.repack_module_input(input)
            namespace = (module, name, input_list, input)
            self.namespace_.append(namespace)
            if not self.trace_module_through(module):
                return tuple(input_list)

        # Identify input to the whole model
        def model_input(module, input):
            if module is model:
                input_list = self.repack_module_input(input)
                self.trace_.model_input_list = input_list

        for mod in model.modules():
            if not isinstance(mod, torch.jit.ScriptModule):
                self.hook_handles.append(mod.register_forward_pre_hook(enter_namespace))
                self.hook_handles.append(mod.register_forward_pre_hook(model_input))

    def _register_forward_hooks(self, model):

        # input is possibly the Tracer
        def exit_namespace(module, input, output):
            assert self.namespace_[-1][0] is module  # check module is on the stack
            for trace_elem in self.trace_.trace_elem_list:
                if trace_elem.module_context_list[-1] is module:
                    if trace_elem.module_output is None:
                        output = self.repack_value(output)
                        trace_elem.module_output = output
            namespace = self.namespace_.pop()
            orig_input = namespace[3]
            # is_tracins is to check that I'm not in an allowed module inside a blocked module
            if self.is_tracing(orig_input) and not self.trace_module_through(module):
                args = namespace[2]
                module_name = namespace[1]
                inplace = self.is_inplace_function(output, args, {})
                output = self.update_inplace_output(inplace, output)
                output = self.update_trace(module, FnType.MODULE, tuple(args), {}, output)
                self.trace_.trace_elem_list[-1].module_fn_name = module_name
                if inplace:
                    assert isinstance(orig_input, tuple) and len(orig_input) == 1
                    input_tracer = orig_input[0]
                    # self is the tracer from when the hook was registered, not when the input
                    # is passed. it doesn't make a difference if the operation is not in place,
                    # but if the operation is in place i need to update the value_ of the tracer
                    # that was passed as input
                    return input_tracer.epilogue(True, output)
                else:
                    return self.epilogue(False, output)

        # Identify output of the whole model
        def model_output(module, input, output):
            if module is model:
                self.trace_.model_output_list = self.repack_model_output(output)
                return self.repack_value(output)

        for mod in model.modules():
            if not isinstance(mod, torch.jit.ScriptModule):
                self.hook_handles.append(mod.register_forward_hook(exit_namespace))
                self.hook_handles.append(mod.register_forward_hook(model_output))

    def move_torch_args_to_kwargs(self, fn, fn_args, fn_kwargs):
        fn_stub = TORCH_FN_OVERRIDE_DICT[fn]
        fn_kwargs = getcallargs(fn_stub, *fn_args, **fn_kwargs)
        return [], fn_kwargs

    def update_inplace_output(self, inplace, out):
        if inplace and isinstance(out, Tensor):
            return out.clone()
        elif inplace:
            raise RuntimeError(f"Inplace function on type {type(out)} not supported")
        else:
            return out

    def __torch_function__(self, fn, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        args, kwargs = self.move_torch_args_to_kwargs(fn, args, kwargs)
        args, kwargs = self.repack_args_kwargs(args, kwargs)
        # get rid of out=None early on since it's often problematic
        if 'out' in kwargs and kwargs['out'] is None: del kwargs['out']
        out = fn(*args, **kwargs)
        inplace = self.is_inplace_function(out, args, kwargs)
        out = self.update_inplace_output(inplace, out)
        out = self.update_trace(fn, FnType.FUNCTION, args, kwargs, out)
        return self.epilogue(inplace, out)

    def update_trace(self, fn, fn_type, fn_args, fn_kwargs, fn_out):
        modules, m_names, m_inputs, _ = zip(*self.namespace_)
        m_names = flatten(m_names)
        # remove first empty name
        m_names = m_names[1:]
        # take only innermost input list, corresponding to modules[-1]
        m_input_list = m_inputs[-1]
        # empty fn_args means it would be an empty tuple, which we don't want
        fn_args = [] if fn_args == () else fn_args
        # repack fn_out
        fn_out = self.repack_value(fn_out, check_already_in_map=False)
        # assign index to fn inputs and outputs
        fn_args_index = [self.trace_.index_from_val(a) for a in fn_args]
        fn_kwargs_index = {k: self.trace_.index_from_val(a) for k,a in fn_kwargs.items()}
        fn_out_index = self.trace_.index_from_val(fn_out, False)
        # generate trace element
        trace_elem = TraceElem(
            fn=fn,
            fn_type=fn_type,
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
            fn_out=fn_out,
            fn_args_index=fn_args_index,
            fn_kwargs_index=fn_kwargs_index,
            fn_out_index=fn_out_index,
            module_context_list=modules,
            prefix_list=m_names,
            module_input_list=m_input_list)
        # update trace
        self.trace_.trace_elem_list.append(trace_elem)
        # return fn_out with newly interned values
        return fn_out

    # https://stackoverflow.com/questions/3024925/create-a-with-block-on-several-context-managers
    def _trace_with_patches(self, model):
        with ExitStack() as stack:
            for mgr in PATCHES:
                stack.enter_context(mgr)
            model(self)

    def trace_model(self, model: Module, wrap_torchscript=True):
        model = copy(model)
        if wrap_torchscript:
            torchscript_wrapper(model)
        self._register_forward_pre_hooks(model)
        self._register_forward_hooks(model)
        self._trace_with_patches(model)
        self.trace_.named_buffers = dict(model.named_buffers())
        self.trace_.named_params = dict(model.named_parameters())
        for h in self.hook_handles:
            h.remove()
        return self.trace_
