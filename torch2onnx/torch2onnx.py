#!/usr/bin/env python
# -*- coding:utf-8 -*-
########################################
# to do list:
# 1. normal model type: import all class from python files
# 2. bad case: resnet34 failed
# 3. simplify argument name
#########################################
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
INVALID_PARAM_ERROR = 'Parameter Invalid Error.'
FAILURE_OPERATION_ERR0R = 'Failure Operation Error.'


# arguments
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Convertor Tool')
    # cuda
    parser.add_argument('--cuda', default=True,
                        help='Using CUDA or not (default: True)')
    # model relates
    # !!!required!!!
    parser.add_argument('--src-model-path', type=str, default='',
                        help='Source Model\'s filename, Notice should be absolut path (default: \'\')')
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
    parser.add_argument('-f', '--net-file', type=str, default="",
                        help='Torch Net Class Name (default: \'\')')
    parser.add_argument('-c', '--cls-name', type=str, default="",
                        help='Torch Net Class Name (default: \'\')')
    parser.add_argument('--func-name', type=str, default="",
                        help='Function Name of getting Net Class Instance (default: \'\')')

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
    return parser.parse_args()


def _parse_python_path(path):
    if not str.endswith(path, '.py'):
        print("{} Not python file, pls check \'{}\'".format(INVALID_PARAM_ERROR, path))
        return False, None, None
    if not os.path.exists(path):
        print("{} No such python file, pls check \'{}\'".format(NOT_FOUND_ERROR, path))
        return False, None, None
    module_path = os.path.dirname(path)
    module_name = str.split(os.path.basename(path), '.py')[0]
    if module_name == '':
        print("{} Invalid .py file, empty filename, pls check \'{}\'".format(INVALID_PARAM_ERROR, path))
        return False, None, None
    return True, module_path, module_name


# variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()


def _import_module(module_path: str, module_name: str):
    if module_path is not None:
        sys.path.append(module_path)

    if module_name is not None:
        try:
            module = importlib.import_module(module_name)
            return True, module
        except Exception as e:
            print('{} Import module \'{}\' failed, please check python file \'{}.{}.py\'. Error: {}'.
                  format(FAILURE_OPERATION_ERR0R, module_name, module_path, module_name, e))
            return False, None


def import_element_by_module(module, elms: list):
    if module is None:
        return False, module
    try:
        globals().update({k: getattr(module, k) for k in elms})
    except Exception as e:
        print('{} Import \'{}\' failed, pls check cls-name \'{}\' and net-file \'{}\'. Error: {}'.
              format(FAILURE_OPERATION_ERR0R, elms, args.cls_name, args.net_file, e))
        return False, module
    return True, module


def import_module_by_python_file(path):
    ok, module_path, module_name = _parse_python_path(path=path)
    if ok:
        return _import_module(module_path=module_path,
                              module_name=module_name)


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
def to_torch_script(model, export_dir: str, export_file: str, batch_size: int, input_shape: list, trace: bool) -> bool:
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


def _load_model_with_torch_script(path: str):
    try:
        model = torch.jit.load(path).to(device)
    except Exception as e:
        return False, None, None
    return True, model, TorchModelType.TORCH_SCRIPT


def _load_model_with_normal_model(path: str):
    try:
        model = torch.load(path)
        if isinstance(model, collections.OrderedDict):
            return False, None, None
        model = model.to(device)
        return True, model, TorchModelType.NORMAL_MODEL
    except Exception as e:
        print("{} Failed to call torch.load(\'{}\'). Error: {}".format(FAILURE_OPERATION_ERR0R, path, e))
        return False, None, None


def _load_model_with_state_dict(path: str, module, class_name: str, func_name: str):
    param = torch.load(path)
    if not isinstance(param, collections.OrderedDict):
        return False, None, None
    if class_name == "" and func_name == "":
        print("{} Torch-Checkpoint(state_dict) format model, "
              "Class Name of net definition or Get Net Instance Function must be provided, "
              "please \'-h\' to see usage.".format(INVALID_PARAM_ERROR))
        return False, None, None
    # try function loader
    elif func_name != "":
        try:
            func = eval(func_name)
            model = func().to(device)
            model.load_state_dict(param)
            return True, model, TorchModelType.STAT_DICT
        except Exception as e:
            print("{} Failed to call \'load_state_dict()\' or \'{}()\'. Error: {}".
                  format(FAILURE_OPERATION_ERR0R, func_name, e))
            pass
    else:
        try:
            cls = getattr(module, class_name)
            model = cls().to(device)
            model.load_state_dict(param)
            return True, model, TorchModelType.STAT_DICT
        except Exception as e:
            print("{} Failed to call \'load_state_dict()\' or \'{}()\'. Error: {}".
                  format(FAILURE_OPERATION_ERR0R, class_name, e))
            return False, None, None


def load_model(model_path, python_path, class_name, func_name):
    # try 1. torch-script?
    ok, model, model_type = _load_model_with_torch_script(path=model_path)
    if ok:
        return ok, model, model_type
    if python_path == "" or class_name == "":
        print("{} Not a Torch-Script format model, python file and class name of net definition must be provided, "
              "please \'-h\' to see usage".
              format(INVALID_PARAM_ERROR))
        return False, None, None
    ok, module = import_module_by_python_file(path=python_path)
    if not ok or module is None:
        print("{} Not a Torch-Script format model, python file with net definition must be provided, "
              "but {} is not a valid python file".format(INVALID_PARAM_ERROR, python_path))
        return False, None, None
    elms = [class_name]
    if func_name != "":
        elms.append(func_name)
    ok, module = import_element_by_module(module=module, elms=elms)
    if not ok:
        print("{} Not a Torch-Script format model, class name of net definition must be provided, "
              "but {} is not the correct class name.".format(INVALID_PARAM_ERROR, class_name))
        return False, None, None
    # try 2. torch-model?
    ok, model, model_type = _load_model_with_normal_model(path=model_path)
    if ok:
        return ok, model, model_type
    # try 3. torch-state-dict?
    ok, model, model_type = _load_model_with_state_dict(path=model_path, module=module,
                                                        class_name=class_name, func_name=func_name)
    if ok:
        return ok, model, model_type
    print("{} Not Invalid Model, please check your model{}".format(FAILURE_OPERATION_ERR0R, model_path))
    return False, None, None


def main():
    # !!!required!!!
    if args.src_model_path == "":
        print("{} Miss necessary parameters, please use \'-h\' to check usage".format(INVALID_PARAM_ERROR))
        return

    # load model
    ok, model, model_type = load_model(model_path=args.src_model_path, python_path=args.net_file,
                                       class_name=args.cls_name, func_name=args.func_name)
    if not ok:
        return
    # prepare to save model
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
    main()
