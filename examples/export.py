import argparse
import os
import random
import configparser
import collections

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from models import *

SEED = 123456

models = {'quant_mobilenet_v1': quant_mobilenet_v1}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model-cfg', required=True, type=str, help='Path to pretrained model .ini configuration file')
parser.add_argument('--output-dir', required=True, type=str, help='Path to export files')
parser.add_argument('--input-npy', required=True, type=str, help='Path to input npy')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    assert os.path.exists(args.model_cfg)
    cfg = configparser.ConfigParser()
    cfg.read(args.model_cfg)
    arch = cfg.get('MODEL', 'ARCH')
    model = models[arch](cfg)
    inp = np.load(args.input_npy)
    export(inp, model, args)


def export(inp, model, args):
    inp = torch.from_numpy(inp).float()
    weight_list, threshold_list, config_list, int_input_list, int_acc_list = model.export(inp)
    exp_dicts = [dict(weight_list), dict(threshold_list)]
    output_dict = collections.defaultdict(list)
    for d in exp_dicts:
        for k, v in d.items():
            output_dict[k].append(v)
    for name_prefix, var_list in output_dict.items():
        with open(os.path.join(args.output_dir, '{}.h'.format(name_prefix)), 'w') as f:
            for var in var_list:
                f.write(var)
                f.write('\n')
    with open(os.path.join(args.output_dir, 'config.h'), 'w') as f:
        for config_line in config_list:
            f.write(config_line)
    for array_tuple in int_input_list:
        np.save(os.path.join(args.output_dir, array_tuple[0] + '.npy'), array_tuple[1])
    for array_tuple in int_acc_list:
        np.save(os.path.join(args.output_dir, array_tuple[0] + '.npy'), array_tuple[1])
    return

if __name__ == '__main__':
    main()
