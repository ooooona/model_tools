#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function

import argparse
import collections
import importlib
import os
import sys
from enum import Enum

import torch


# torch model type
class TorchModelType(Enum):
    STAT_DICT = 1
    NORMAL_MODEL = 2
    TORCH_SCRIPT = 3


# error
NOT_FOUND_ERROR = 'Not Found Error.'
PARAM_INVALID_ERROR = 'Parameter Invalid Error.'
FAILURE_OPERATION_ERR0R = 'Failure Operation Error.'

# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# arguments
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Convertor Tool')
    # model relates
    parser.add_argument('--src-model-path', type=str, default="./model.pth",
                        help='Source Model\'s filename, Notice should be absolut path (default: \'./model.pth\')')
    parser.add_argument('--dst-model-dir', type=str, default="./",
                        help='Destination Model\'s directory (default: \'./\')')
    parser.add_argument('--dst-model-name', type=str, default="dst_model",
                        help='Destination Model\'s filename, Notice without postfix (default: \'dst_model\')')
    # save
    parser.add_argument('-o', '--onnx', default=True,
                        help='Saving the current Model as ONNX (default: True)')
    parser.add_argument('-t', '--torch-script', default=True,
                        help='Saving the current Model as TORCH-SCRIPT (default: True)')
    # net
    parser.add_argument('-f', '--net-file', type=str, default="./net.py",
                        help='Torch Net Class Name (default: \'./net.py\')')
    parser.add_argument('-c', '--cls-name', type=str, default="Net",
                        help='Torch Net Class Name (default: \'Net\')')
    parser.add_argument('--func-name', type=str, default="GetNetInstance",
                        help='Function Name of getting Net Class Instance (default: \'./GetNetInstance\')')

    parser.add_argument('-b', '--batch', type=int, default='1',
                        help='Input Name for Model (default: 1)')
    # input
    parser.add_argument('--input-name', type=str, default='input',
                        help='Input Name for Model (default: \'input\')')
    parser.add_argument('--input-shape', type=int, nargs='+', default=(1, 28, 28),
                        help='Input Shape for Model, Notice without Batch-Size! (default: (1, 28, 28))')
    # output
    parser.add_argument('--output-name', type=str, default='output',
                        help='Output Name for Model (default: \'output\')')
    parser.add_argument('--output-shape', type=int, nargs='+', default=10,
                        help='Output Shape for Model, Notice without Batch-Size! (default: (10))')

    # parser.add_argument('--onnx', default=True,
    #                     help='Saving the current Model as ONNX (default: True)')
    # parser.add_argument('--torch-script', default=True,
    #                     help='Saving the current Model as TORCH-SCRIPT (default: True)')

    # parser.add_argument('--net-file-path', type=str, default="./net.py",
    #                     help='Torch Net Class Name (default: \'./net.py\')')
    # parser.add_argument('--net-cls-name', type=str, default="Net",
    #                     help='Torch Net Class Name (default: \'Net\')')
    # parser.add_argument('--ext-func-name', type=str, default="GetNetInstance",
    #                     help='Function Name of getting Net Class Instance (default: \'./GetNetInstance\')')

    # parser.add_argument('--batch-size', type=int, default=1,
    #                     help='Batch Size for Model (default: 1)')
    # parser.add_argument('--input-name', type=str, default="input",
    #                     help='Input Name for Model (default: input)')
    # parser.add_argument('--input-shape', type=int, nargs='+', default=(1, 28, 28),
    #                     help='Input Shape for Model, Notice without Batch-Size! (default: (1, 28, 28))')
    #
    # parser.add_argument('--output-name', type=str, default="output",
    #                     help='Output Name for Model (default: \'output\')')
    # parser.add_argument('--output-shape', type=int, nargs='+', default=10,
    #                     help='Output Shape for Model, Notice without Batch-Size! (default: (10))')
    return parser.parse_args()


def parse_net_cls_path(net_file_path):
    if not str.endswith(net_file_path, '.py'):
        print("{} Not .py file, pls check \'{}\'".format(PARAM_INVALID_ERROR, net_file_path))
        return False, None, None
    if not os.path.exists(net_file_path):
        print("{} No such file, pls check \'{}\'".format(NOT_FOUND_ERROR, net_file_path))
        return False, None, None
    import_module_dir = os.path.dirname(net_file_path)
    import_module_name = str.split(os.path.basename(net_file_path), '.py')[0]
    if import_module_name == '':
        print("{} Invalid .py file, empty filename, pls check \'{}\'".format(PARAM_INVALID_ERROR, net_file_path))
        return False, None, None
    return True, import_module_dir, import_module_name


def import_module(module_path: str, module_name: str, module_elm: list):
    try:
        sys.path.append(module_path)
        module = importlib.import_module(module_name)
        globals().update({k: getattr(module, k) for k in module_elm})
        return module
    except Exception as e:
        print('{} Import failed, pls check cls-name \'{}\' and net-file \'{}\'. Error: {}'.
              format(FAILURE_OPERATION_ERR0R, args.cls_name, args.net_file, e))
        return None


# torch to onnx
def to_onnx(model, export_dir: str, export_file: str, batch_size: int,
            input_names: list, input_shape: tuple,
            output_names: list) -> bool:
    # set the model to inference mode
    model.eval()

    # export filename
    export_file = os.path.join(export_dir, "%s.onnx" % export_file)
    # input tensor
    if type(input_shape) is not tuple and type(input_shape) is not list:
        input_tensor = torch.randn(batch_size, input_shape).to(device)
    else:
        input_tensor = torch.randn(batch_size, *input_shape).to(device)
    # output tensor
    try:
        output_tensor = model(input_tensor)
    except Exception as e:
        print("Failed to convert to OONX, err: {}".format(e))
        return False
    # dynamic_axes
    dynamic_axes = dict()
    for input_name in input_names:
        dynamic_axes[input_name] = {0: "-1"}
    for output_name in output_names:
        dynamic_axes[output_name] = {0: "-1"}
    # export to onnx
    try:
        torch.onnx.export(model,
                          input_tensor,
                          export_file,
                          input_names=input_names,
                          output_names=output_names,
                          example_outputs=output_tensor,
                          dynamic_axes=dynamic_axes
                          )
    except Exception as e:
        print("Failed to convert to OONX, err: {}".format(e))
        return False
    else:
        print("Success to convert to OONX, dump to \"{}\"".format(export_file))
        return True


# torch to torch-script
def to_torch_script(model, export_dir: str, export_file: str, batch_size: int, input_shape: list, trace=True) -> bool:
    # set the model to inference mode
    model.eval()

    # export filename
    export_file = os.path.join(export_dir, "%s.pth" % export_file)
    script_model = model
    if trace:
        if type(input_shape) is not tuple and type(input_shape) is not list:
            input_tensor = torch.randn(batch_size, input_shape).to(device)
        else:
            input_tensor = torch.randn(batch_size, *input_shape).to(device)
        script_model = torch.jit.trace(func=model, example_inputs=input_tensor)
    try:
        torch.jit.save(m=script_model, f=export_file)
    except Exception as e:
        print("{} Failed to convert to TORCH-SCRIPT. Error: {}".format(FAILURE_OPERATION_ERR0R, e))
        return False
    else:
        print("Success to convert to TORCH-SCRIPT, dump to \"{}\"".format(export_file))
        return True


def _load_model_with_state_dict(param, cls=None, func=None):
    model = None
    if func is not None:
        try:
            model = func()
            model.load_state_dict(param)
        except Exception as e:
            print('{} Failed to call load_state_dict() or {}. Error: {}'.format(FAILURE_OPERATION_ERR0R, func, e))
            pass
    if model is None and cls is not None:
        try:
            model = cls()
            model.load_state_dict(param)
        except Exception as e:
            print('{} Failed to call load_state_dict() or {}. Error: {}'.format(FAILURE_OPERATION_ERR0R, cls, e))
            return False, None, None

    return True, model, TorchModelType.STAT_DICT


def _load_model_with_normal_model(model):
    return True, model, TorchModelType.NORMAL_MODEL


def _load_model_with_torch_script(path):
    try:
        model = torch.jit.load(path)
    except Exception:
        return False, None, None
    return True, model, TorchModelType.TORCH_SCRIPT


def _load_model(model, net_type, path, cls=None, func=None):
    if isinstance(model, collections.OrderedDict):
        return _load_model_with_state_dict(param=model, cls=cls, func=func)
    elif isinstance(model, net_type):
        return _load_model_with_normal_model(model=model)
    else:
        return _load_model_with_torch_script(path=path)


def load_model(path, cls=None, func=None):
    try:
        ok = False
        model_type = None
        model = torch.load(path)
        if func is not None:
            ok, model, model_type = _load_model(model=model, net_type=type(func()), path=path, cls=None, func=func)
        if not ok and cls is not None:
            ok, model, model_type = _load_model(model=model, net_type=cls, path=path, cls=cls, func=None)
        if not ok:
            return False, None, None
        else:
            return ok, model, model_type
    except Exception as e:
        print("{} torch.load(\'{}\') failed. Error: {}".format(FAILURE_OPERATION_ERR0R, path, e))
        return False, None, None


def main(module):
    func = None
    try:
        func = eval(args.func_name)
        print("Use function \'{}\' in {}".format(args.func_name, args.net_file))
    except Exception as e:
        # print("{} No such function \'{}\' in {}".format(NOT_FOUND_ERROR, args.func_name, args.net_file))
        pass
    cls = None
    try:
        cls = getattr(module, args.cls_name)
        if cls is None:
            print(
                "{} No such class \'{}\', pls check {}.".format(NOT_FOUND_ERROR, args.cls_name, args.net_file))
    except Exception as e:
        print("{} No such class \'{}\', pls check {}. Error: {}".
              format(NOT_FOUND_ERROR, args.cls_name, args.net_file, e))
        return
    # load model
    ok, model, model_type = load_model(path=args.src_model_path, cls=cls, func=func)
    if not ok:
        return
    if not os.path.exists(args.dst_model_dir) or not os.path.isdir(args.dst_model_dir):
        os.makedirs(args.dst_model_dir)
    # convert to onnx
    if args.onnx:
        to_onnx(model=model, export_dir=args.dst_model_dir, export_file=args.dst_model_name,
                batch_size=args.batch, input_names=[args.input_name], input_shape=args.input_shape,
                output_names=[args.output_name])
    # convert to torch-script
    if args.torch_script:
        if model_type == TorchModelType.TORCH_SCRIPT:
            to_torch_script(model=model, export_dir=args.dst_model_dir, export_file=args.dst_model_name,
                            batch_size=args.batch, input_shape=args.input_shape, trace=False)
        else:
            to_torch_script(model=model, export_dir=args.dst_model_dir, export_file=args.dst_model_name,
                            batch_size=args.batch, input_shape=args.input_shape, trace=True)


if __name__ == '__main__':
    args = get_args()
    ok, import_module_dir, import_module_name = parse_net_cls_path(net_file_path=args.net_file)
    if ok:
        module = import_module(module_path=import_module_dir,
                               module_name=import_module_name,
                               module_elm=[args.cls_name])
        if module is not None:
            main(module=module)
