### PyTorch Model Convert Tool

Convert from PyTorch to ONNX and TorchScript

Support source Type:
- torch checkpoint
- torch model
- torch script

Dependency:
- torch

Usage: 
```
torch2onnx.py [-h] [--src-model-path SRC_MODEL_PATH] [--dst-model-dir DST_MODEL_DIR] [--dst-model-name DST_MODEL_NAME] [-o ONNX] [-t TORCH_SCRIPT] [-f NET_FILE] [-c CLS_NAME] [--func-name FUNC_NAME] [-b BATCH] [--input-name INPUT_NAME]
                     [--input-shape INPUT_SHAPE [INPUT_SHAPE ...]] [--output-name OUTPUT_NAME] [--output-shape OUTPUT_SHAPE [OUTPUT_SHAPE ...]]


optional arguments:
  -h, --help            show this help message and exit
  --src-model-path SRC_MODEL_PATH
                        Source Model's filename, Notice should be absolut path (default: './model.pth')
  --dst-model-dir DST_MODEL_DIR
                        Destination Model's directory (default: './')
  --dst-model-name DST_MODEL_NAME
                        Destination Model's filename, Notice without postfix (default: 'dst_model')
  -o ONNX, --onnx ONNX  Saving the current Model as ONNX (default: True)
  -t TORCH_SCRIPT, --torch-script TORCH_SCRIPT
                        Saving the current Model as TORCH-SCRIPT (default: True)
  -f NET_FILE, --net-file NET_FILE
                        Torch Net Class Name (default: './net.py')
  -c CLS_NAME, --cls-name CLS_NAME
                        Torch Net Class Name (default: 'Net')
  --func-name FUNC_NAME
                        Function Name of getting Net Class Instance (default: './GetNetInstance')
  -b BATCH, --batch BATCH
                        Input Name for Model (default: 1)
  --input-name INPUT_NAME
                        Input Name for Model (default: 'input')
  --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input Shape for Model, Notice without Batch-Size! (default: (1, 28, 28))
  --output-name OUTPUT_NAME
                        Output Name for Model (default: 'output')
  --output-shape OUTPUT_SHAPE [OUTPUT_SHAPE ...]
                        Output Shape for Model, Notice without Batch-Size! (default: (10))
```