"""
python convert_to_onnx.py --input_model "/Users/manali/nyu/COURSES/Sem4/MLOps/ECE-GY-9183-Machine-Learning-Systems-Engg-Operations-Project/src/serving/inference_service/model.pth" --output_model "model_regnet_2.onnx"
"""

import os
import argparse
import torch
import onnx

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format.")
parser.add_argument('--input_model', type=str, required=True, help='Path to the input PyTorch model (.pth)')
parser.add_argument('--output_model', type=str, required=True, help='Path to save the output ONNX model (.onnx)')
args = parser.parse_args()

input_model_path = args.input_model
onnx_model_path = args.output_model

device = torch.device("cpu")

# Load the PyTorch model
model = torch.load(input_model_path, map_location=device, weights_only=False)

# Dummy input - used to clarify the input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path,
    export_params=True, 
    opset_version=20,
    do_constant_folding=True, 
    input_names=['input'],
    output_names=['output'], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"ONNX model saved to {onnx_model_path}")

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("Converted to ONNX model.")