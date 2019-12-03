from .mobilenetv1 import *
from .proxylessnas import *
from .vgg import *

models_dict = {'quant_mobilenet_v1': quant_mobilenet_v1,
               'quant_proxylessnas_mobile14': quant_proxylessnas_mobile14,
               'quant_proxylessnas_cpu': quant_proxylessnas_cpu,
               'quant_proxylessnas_mobile': quant_proxylessnas_mobile,
               'quant_proxylessnas_gpu': quant_proxylessnas_gpu}
