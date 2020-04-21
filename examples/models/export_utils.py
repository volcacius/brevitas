import numpy as np
import torch
from functools import reduce
from operator import mul


def is_depthwise(conv):
    return conv.in_channels == conv.out_channels == conv.groups


def prune_weight_matrix(weight_matrix, in_ch_pruning_mask, out_ch_pruning_mask, is_depthwise):
    if is_depthwise:
        if out_ch_pruning_mask is not None and in_ch_pruning_mask is not None:
            weight_matrix = weight_matrix[np.logical_and(out_ch_pruning_mask, in_ch_pruning_mask)]
        elif out_ch_pruning_mask is None and in_ch_pruning_mask is not None:
            weight_matrix = weight_matrix[in_ch_pruning_mask]
        elif out_ch_pruning_mask is not None and in_ch_pruning_mask is None:
            weight_matrix = weight_matrix[out_ch_pruning_mask]
    else:
        if out_ch_pruning_mask is not None and in_ch_pruning_mask is not None:
            weight_matrix = weight_matrix[out_ch_pruning_mask][:, in_ch_pruning_mask]
        elif out_ch_pruning_mask is None and in_ch_pruning_mask is not None:
            weight_matrix = weight_matrix[:, in_ch_pruning_mask]
        elif out_ch_pruning_mask is not None and in_ch_pruning_mask is None:
            weight_matrix = weight_matrix[out_ch_pruning_mask]
    return weight_matrix


def hls_weight_matrix_conv(conv, in_ch_pruning_mask=None, out_ch_pruning_mask=None, sign_factor=None):
    weight_matrix = conv.int_weight.detach().cpu().numpy()
    weight_matrix = prune_weight_matrix(weight_matrix, in_ch_pruning_mask, out_ch_pruning_mask, is_depthwise(conv))
    transpose_axes = (0, 3, 2, 1)
    weight_matrix = np.transpose(weight_matrix, axes=transpose_axes)
    weight_matrix = np.rot90(weight_matrix, axes=(1, 2))
    weight_matrix = np.flip(weight_matrix, axis=1)  # flip along input channel
    if sign_factor is not None:
        sign_factor = sign_factor.int().detach().cpu().numpy()
        if out_ch_pruning_mask is not None:
            sign_factor = sign_factor[out_ch_pruning_mask]
        weight_matrix = weight_matrix * np.reshape(sign_factor, newshape=[-1, 1, 1, 1])
    weight_matrix = weight_matrix.astype('object')
    return weight_matrix


def weight_matrix_conv_pruning_mask(conv):
    weight_matrix = conv.int_weight.detach().cpu().numpy()
    weight_matrix = np.reshape(weight_matrix, newshape=[weight_matrix.shape[0], -1])
    out_ch_pruning_mask = weight_matrix.any(axis=1)
    return out_ch_pruning_mask


def hls_weight_matrix_fc(fc, in_ch_pruning_mask):
    weight_matrix = fc.int_weight.detach().cpu().numpy()
    weight_matrix = prune_weight_matrix(weight_matrix, in_ch_pruning_mask, None, False)
    weight_matrix = weight_matrix.astype('object')
    return weight_matrix


def hls_weight_matrix_simd_pe(weight_matrix, bit_width, simd, pe):
    matrix_height = weight_matrix.shape[0]
    matrix_width = reduce(mul, weight_matrix.shape[1:], 1)
    assert matrix_width % simd == 0
    assert matrix_height % pe == 0
    weight_matrix = weight_matrix.reshape((matrix_height, matrix_width // simd, simd))
    simd_pe_shape = (pe, matrix_height * matrix_width // (simd * pe))
    # object is required for arbitrary large ints
    weight_matrix_simd_pe = np.zeros(shape=simd_pe_shape, dtype='object')
    for i in range(0, matrix_height):
        target_pe = i % pe
        offset_pe = int(i // pe) * matrix_width // simd
        for j in range(0, matrix_width // simd):
            val = pack_array(weight_matrix[i, j, :], bit_width)
            weight_matrix_simd_pe[target_pe, offset_pe + j - matrix_height * matrix_width // (simd * pe)] = val
    weight_matrix_simd_pe = np.expand_dims(weight_matrix_simd_pe, axis=0)
    return weight_matrix_simd_pe


def hls_weight_string_conv(conv, hls_var_name, weight_bit_width, sign_factor, in_ch_pruning_mask=None, out_ch_pruning_mask=None, simd=None, pe=None, hex_repr=True):
    weight_matrix = hls_weight_matrix_conv(conv, in_ch_pruning_mask, out_ch_pruning_mask, sign_factor)
    if is_depthwise(conv):
        simd = simd if simd is not None else 1
        pe = pe if pe is not None else weight_matrix.shape[0]  # output channels
    else:
        simd = simd if simd is not None else weight_matrix.shape[1]  # input channels
        pe = pe if pe is not None else weight_matrix.shape[0]  # output channels
    weight_matrix_simd_pe = hls_weight_matrix_simd_pe(weight_matrix, weight_bit_width, simd, pe)
    matrix_string = hls_matrix_string(weight_matrix_simd_pe, signature_style='weights', hex_repr=hex_repr)
    matrix_string = matrix_string.format(
        bit_width=weight_bit_width,
        hls_var_name=hls_var_name,
        simd=simd,
        pe=pe,
        tile=tile_conv(conv, simd, pe))
    return matrix_string


def hls_bias_string_fc(int_bias, hls_var_name, output_bit_width, bias_bit_width, pe=None, hex_repr=False):
    pe = pe if pe is not None else len(int_bias.view(-1))
    bias_matrix = int_bias.view(-1, 1).cpu().numpy()
    matrix_height = bias_matrix.shape[0]
    bias_matrix_pe = hls_bias_matrix_pe(
        bias_matrix=bias_matrix,
        bias_bit_width=bias_bit_width,
        pe=pe,
        matrix_height=matrix_height,
        pack=False)
    bias_matrix_pe = np.expand_dims(bias_matrix_pe, axis=0)
    matrix_string = hls_matrix_string(bias_matrix_pe, signature_style='bias', hex_repr=hex_repr)
    matrix_string = matrix_string.format(
        neuron_folding=matrix_height // pe,
        pe=pe,
        bias_bit_width=bias_bit_width,
        output_bit_width=output_bit_width,
        hls_var_name=hls_var_name)
    return matrix_string


def hls_weight_string_fc(fc, hls_var_name, weight_bit_width, in_ch_pruning_mask=None, simd=None, pe=None, hex_repr=True):
    simd = simd if simd is not None else fc.in_features
    pe = pe if pe is not None else fc.out_features
    weight_matrix = hls_weight_matrix_fc(fc, in_ch_pruning_mask)
    weight_matrix_simd_pe = hls_weight_matrix_simd_pe(weight_matrix, weight_bit_width, simd, pe)
    matrix_string = hls_matrix_string(weight_matrix_simd_pe, signature_style='weights', hex_repr=hex_repr)
    matrix_string = matrix_string.format(
        bit_width=weight_bit_width,
        hls_var_name=hls_var_name,
        simd=simd,
        pe=pe,
        tile=tile_fc(fc, simd, pe))
    return matrix_string


def hls_config_string_conv(conv, name, weight_bit_width, output_bit_width, input_2d_shape, output_2d_shape, in_ch_pruning_mask, out_ch_pruning_mask, simd=None, pe=None):
    in_ch = np.count_nonzero(in_ch_pruning_mask) if in_ch_pruning_mask is not None else conv.in_channels
    out_ch = np.count_nonzero(out_ch_pruning_mask) if out_ch_pruning_mask is not None else conv.out_channels
    if is_depthwise(conv):
        simd = simd if simd is not None else 1
        pe = pe if pe is not None else out_ch
    else:
        simd = simd if simd is not None else in_ch
        pe = pe if pe is not None else out_ch
    config_string_list = []
    config_string_list.append(define('SIMD_{}'.format(name), simd))
    config_string_list.append(define('PE_{}'.format(name), pe))
    config_string_list.append(define('TILE_{}'.format(name), tile_conv(conv, simd, pe)))
    config_string_list.append(define('WIDTH_{}'.format(name), weight_bit_width))
    config_string_list.append(define('KERNEL_DIM_{}'.format(name), conv.kernel_size[0]))
    config_string_list.append(define('IFM_DIM_{}'.format(name), input_2d_shape[0]))
    config_string_list.append(define('OFM_DIM_{}'.format(name), output_2d_shape[0]))
    config_string_list.append(define('IFM_CHANNELS_{}'.format(name), in_ch))
    config_string_list.append(define('OFM_CHANNELS_{}'.format(name), out_ch))
    config_string_list.append(define('STRIDE_{}'.format(name), conv.stride[0]))
    config_string_list.append(define('PADDING_{}'.format(name), conv.padding[0]))
    config_string_list.append(define('OUTPUT_WIDTH_{}'.format(name), output_bit_width))
    config_string_list.append('\n')
    return ''.join(config_string_list)


def hls_config_string_fc(fc, name, weight_bit_width, output_bit_width, in_ch_pruning_mask=None, simd=None, pe=None):
    in_ch = np.count_nonzero(in_ch_pruning_mask) if in_ch_pruning_mask is not None else fc.in_features
    simd = simd if simd is not None else in_ch
    pe = pe if pe is not None else fc.out_features
    config_string_list = []
    config_string_list.append(define('SIMD_{}'.format(name), simd))
    config_string_list.append(define('PE_{}'.format(name), pe))
    config_string_list.append(define('TILE_{}'.format(name), tile_fc(fc, simd, pe)))
    config_string_list.append(define('WIDTH_{}'.format(name), weight_bit_width))
    config_string_list.append(define('IN_FEATURES_{}'.format(name), in_ch))
    config_string_list.append(define('OUT_FEATURES_{}'.format(name), fc.out_features))
    config_string_list.append(define('OUTPUT_WIDTH_{}'.format(name), output_bit_width))
    config_string_list.append('\n')
    return ''.join(config_string_list)


def tile_conv(conv, simd, pe):
    return conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels * conv.out_channels // (simd * pe * conv.groups)


def tile_fc(conv, simd, pe):
    return conv.in_features * conv.out_features // (simd * pe)


def define(name, val):
    return "#define {} {}\n".format(name, val)


def hls_matrix_string(int_x, signature_style, hex_repr=True):
    string_list = [""]
    string_list = hls_matrix_string_signature(string_list, signature_style)
    matrix_to_string_list(int_x, int_x.ndim - 1, string_list, hex_repr)
    decl = ''.join(string_list)
    decl += ";\n"
    return decl


def hls_matrix_string_signature(string_list, signature_style):
    if signature_style == 'weights':
        string_list.append("static FixedPointWeights<{simd}, ap_int<{bit_width}>,{pe},{tile}> {hls_var_name} = ")
        return string_list
    elif signature_style == 'thresholds':
        string_list.append(
            "ThresholdsActivation{postfix}"
            "<{neuron_folding},"
            "{pe},"
            "{num_thresholds},"
            "ap_int<{threshold_bit_width}>,"
            "ap_int<{precision}>,"
            "{starting_value}> {hls_var_name} = ")
        return string_list
    elif signature_style == 'bias':
        string_list.append(
            "ThresholdsActivation"
            "<{neuron_folding},"
            "{pe},"
            "1,"
            "ap_int<{bias_bit_width}>,"
            "ap_int<{output_bit_width}>,"
            "0,"
            "std::plus<ap_int<{output_bit_width}>>> {hls_var_name} = ")
        return string_list
    else:
        raise Exception("Signature style not recognized: {}".format(signature_style))


def matrix_to_string_list(int_x, dims, string_list, hex_repr):
    string_list.append("{{ ")
    for i in range(0, int_x.shape[0]):
        if dims > 0:
            matrix_to_string_list(int_x[i], dims - 1, string_list, hex_repr)
        else:
            if hex_repr:
                val = str(hex(int_x[i]))
                val = "\"{}\"".format(val)
            else:
                val = str(int_x[i])
            string_list.append(val)
        if i < int_x.shape[0] - 1:
            string_list.append(", \n")
    string_list.append("}}")
    return


def pack_array(array, bit_width):
    val = 0
    for i in range(len(array)):
        tmp = array[i]
        if tmp < 0:
            tmp = (2 ** bit_width) + tmp
        tmp = tmp * (2 ** (bit_width * i))
        val += tmp
    return val


def signed_to_unsigned_int(array, bit_width):
    return int(np.binary_repr(array, bit_width), 2)


def scale_bias_fusion(bn, scale_factor_init, bias_factor_init):
    scale_factor = scale_factor_init.view(-1)
    if isinstance(bias_factor_init, torch.Tensor):
        bias_factor = bias_factor_init.view(-1)
    else:
        bias_factor = bias_factor_init
    std_dev = torch.sqrt(bn.running_var + bn.eps).view(-1)
    scale_factor = scale_factor / std_dev
    bias_factor = bias_factor - bn.running_mean.view(-1)
    bias_factor = bias_factor / std_dev
    if bn.affine:
        scale_factor = scale_factor * bn.weight.data.view(-1)
        bias_factor = bias_factor * bn.weight.data.view(-1) + bn.bias.data.view(-1)
    sign_factor = torch.sign(scale_factor)
    scale_factor = torch.abs(scale_factor)
    return scale_factor, bias_factor, sign_factor


def pragma(content):
    return "#pragma {content}\n".format(content=content)


def hls_pragma_interface(content):
    return pragma("HLS INTERFACE {content}".format(content=content))


def include(header):
    return "#include \"{header}\"\n".format(header=header)


def hls_threshold_matrix(
        activation,
        acc_bit_width,
        acc_scale_factor,
        acc_bias_factor,
        enable_pruning,
        out_ch_pruning_mask=None):
    int_input_range = torch.arange(
        start=- 2 ** (acc_bit_width - 1),
        # stop is not included, so last value is stop -1
        end=2 ** (acc_bit_width - 1),
        # all possible int values resulting from accumulation
        dtype=torch.float64, # float32 is not enough for the first layer
        device=acc_scale_factor.device)
    assert (acc_scale_factor >= 0.0).all()
    input_range = torch.ger(acc_scale_factor.type(torch.float64), int_input_range)
    input_range = input_range.type(torch.float32)
    # all possible float values resulting from accumulation
    # and add and mul coefficients from previous layers, per output channel
    input_range = input_range + acc_bias_factor.view(-1, 1)
    # all possible float values resulting from the activation,
    # given the add and mul coefficients from previous layers
    input_range = input_range.view(1, input_range.shape[0], input_range.shape[1], 1)
    output_range, _, output_bit_width = activation(input_range)
    output_range = output_range.view(output_range.shape[1], output_range.shape[2]).squeeze()
    # All possible activation values from the quantization function,
    # without taking into account add and mul coefficients from previous layers
    output_tensor_quant = activation.act_quant_proxy.fused_activation_quant_proxy.tensor_quant
    output_min_int_val = output_tensor_quant.int_quant.min_int(output_bit_width)
    output_max_int_val = output_tensor_quant.int_quant.max_int(output_bit_width)
    output_scale_factor = activation.quant_act_scale().view(-1)
    output_all_int_val = torch.arange(output_min_int_val, output_max_int_val + 1, device=output_scale_factor.device)
    output_all_val = torch.ger(output_scale_factor, output_all_int_val).squeeze()
    # switch to numpy
    output_all_val = output_all_val.detach().cpu().numpy()
    output_range = output_range.detach().cpu().numpy()
    output_bit_width = output_bit_width.int().item()
    unique_index_list = []
    new_out_ch_pruning_mask = []
    # iterate over output channels
    for i in range(output_range.shape[0]):
        unique_value_index = np.unique(output_range[i, :], return_index=True)
        nonzero_ch = np.any(unique_value_index[0])
        new_out_ch_pruning_mask.append(nonzero_ch)
        # get all val per output channel or not
        output_all_val_vec = output_all_val[i, :] if len(output_all_val.shape) == 2 else output_all_val
        min_generated_value = unique_value_index[0][0]
        # first [1] refers to the return value of unique,
        # [1:] refers to the fact that the first threshold is discarded
        unique_index = unique_value_index[1][1:]
        # if the smallest generated value is not the minimum possible,
        # insert the smallest possible threshold as many times as dictated
        # by how big is the smallest value generated by the activation is
        # (by looking at its index in the vector of all possible outputs)
        if not np.isclose(min_generated_value, output_all_val_vec.min()):
            # I expect the condition to return only one value
            # (the index of min_generated_value in output_all_val_vec)
            repeats = np.flatnonzero(np.isclose(output_all_val_vec, min_generated_value)).item()
            # the index for the smallest possible threshold is 0, but -1 is added below
            # so then here it's 1
            unique_index = np.concatenate((np.repeat(1, repeats=repeats), unique_index))
        # if it's still empty, set it to the threshold indeces that sets the smallest possible value
        # (the missing -1 in the index is put below)
        if len(unique_index) == 0:
            unique_index = np.repeat(2 ** acc_bit_width, repeats=2 ** output_bit_width - 1)
        # if some output values are always skipped, I need to repeat some thresholds
        elif len(unique_index) < 2 ** output_bit_width - 1:
            # min non zero val in the output
            output_non_zero_min_val = np.min(output_all_val_vec[np.nonzero(output_all_val_vec)])
            # how many times each threshold should be repeated, starting from 1
            repeats = np.zeros(len(unique_index), dtype=int)
            for j in range(len(unique_index)):
                # some thresholds might already be the same. In that case, keep them as they are
                numer = max(output_non_zero_min_val, (output_range[i, unique_index[j]] - output_range[i, unique_index[j] - 1]))
                repeats[j] += int(round(numer / output_non_zero_min_val))
            extended_unique_index_list = []
            # generate the new index array based on repeats
            for k in range(len(unique_index)):
                for _ in range(repeats[k]):
                    extended_unique_index_list.append(unique_index[k])
            unique_index = np.array(extended_unique_index_list)
            # the missing threshold indeces should be the zero threshold index,
            # without -1 that we are putting later below
            num_missing_thresholds = (2 ** output_bit_width - 1) - len(unique_index)
            for _ in range(num_missing_thresholds):
                unique_index = np.append(unique_index, 2 ** acc_bit_width)
        unique_index_list.append(unique_index)
    unique_index_matrix = np.vstack(tuple(unique_index_list))
    # because thresholding is performed without equality, i.e. val > thr
    threshold_index_matrix = (unique_index_matrix - 1).astype(np.int64)
    # get the corresponding thresholds using fancy indexing
    threshold_matrix = int_input_range.int().cpu().numpy()[threshold_index_matrix]
    # convert pruning list to array
    new_out_ch_pruning_mask = np.array(new_out_ch_pruning_mask)
    if out_ch_pruning_mask is not None:
        new_out_ch_pruning_mask = np.logical_and(new_out_ch_pruning_mask, out_ch_pruning_mask)
    if enable_pruning:
        threshold_matrix = threshold_matrix[new_out_ch_pruning_mask]  # select non pruned channels
    else:
        new_out_ch_pruning_mask = None
    threshold_matrix = threshold_matrix.astype('object')
    return threshold_matrix, new_out_ch_pruning_mask


def hls_bias_matrix_pe(
        bias_matrix,
        matrix_height,
        bias_bit_width,
        pack,
        pe):
    assert len(bias_matrix.shape) == 2
    assert bias_matrix.shape[1] == 1
    assert matrix_height % pe == 0
    shape = (pe, matrix_height // pe) if pack else (pe, matrix_height // pe, 1)
    bias_pe = np.zeros(shape=shape, dtype='object')
    for i in range(0, matrix_height):
        target_pe = i % pe
        offset_pe = i // pe
        val = pack_array(bias_matrix[i], bias_bit_width) if pack else bias_matrix[i]
        bias_pe[target_pe, offset_pe - matrix_height // pe] = val
    return bias_pe


def hls_threshold_matrix_pe(
        thresholds_matrix,
        matrix_height,
        num_thresholds,
        acc_bit_width,
        pe,
        pack):
    assert matrix_height % pe == 0
    shape = (pe, matrix_height // pe) if pack else (pe, matrix_height // pe, num_thresholds)
    thresholds_pe = np.zeros(shape=shape, dtype='object')
    for i in range(0, matrix_height):
        target_pe = i % pe
        offset_pe = i // pe
        val = pack_array(thresholds_matrix[i], acc_bit_width) if pack else thresholds_matrix[i]
        thresholds_pe[target_pe, offset_pe - matrix_height // pe] = val
    return thresholds_pe


def hls_threshold_string(
        activation,
        output_bit_width,
        hls_var_name,
        acc_bit_width,
        acc_scale_factor,
        acc_bias_factor,
        enable_pruning,
        out_ch_pruning_mask=None,
        pe=None,
        starting_value=0,
        pack=False,
        hex_repr=False):
    threshold_matrix, out_ch_pruning_mask = hls_threshold_matrix(
        activation=activation,
        acc_bit_width=acc_bit_width,
        acc_scale_factor=acc_scale_factor,
        acc_bias_factor=acc_bias_factor,
        enable_pruning=enable_pruning,
        out_ch_pruning_mask=out_ch_pruning_mask)
    matrix_height = threshold_matrix.shape[0]
    num_thresholds = threshold_matrix.shape[1]
    pe = pe if pe is not None else matrix_height
    threshold_matrix_pe = hls_threshold_matrix_pe(
        thresholds_matrix=threshold_matrix,
        matrix_height=matrix_height,
        num_thresholds=num_thresholds,
        acc_bit_width=acc_bit_width,
        pe=pe,
        pack=pack)
    threshold_matrix_pe = np.expand_dims(threshold_matrix_pe, axis=0)
    matrix_string = hls_matrix_string(
        threshold_matrix_pe,
        signature_style='thresholds',
        hex_repr=hex_repr)
    matrix_string = matrix_string.format(
        postfix="",
        neuron_folding=matrix_height // pe,
        pe=pe,
        num_thresholds=num_thresholds,
        threshold_bit_width=acc_bit_width,
        precision=output_bit_width,
        starting_value=starting_value,
        hls_var_name=hls_var_name)
    return matrix_string, out_ch_pruning_mask

