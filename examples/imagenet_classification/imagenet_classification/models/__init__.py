from .mobilenetv1 import *
from .efficientnet_edge import *
from .proxylessnas import *
from .vgg import *

models_dict = {'quant_mobilenet_v1': quant_mobilenet_v1,
               'quant_tf_efficientnet_es': quant_tf_efficientnet_es,
               'quant_tf_efficientnet_em': quant_tf_efficientnet_em,
               'quant_tf_efficientnet_el': quant_tf_efficientnet_el,
               'quant_proxylessnas_mobile14': quant_proxylessnas_mobile14,
               'quant_proxylessnas_cpu': quant_proxylessnas_cpu,
               'quant_proxylessnas_mobile': quant_proxylessnas_mobile,
               'quant_proxylessnas_gpu': quant_proxylessnas_gpu,
               'quant_tf_efficientnet_lite2': quant_tf_efficientnet_lite2}

