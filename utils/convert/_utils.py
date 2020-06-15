import torch

def export_onnx(model, export_path, type='prn'):
    input_tensor = torch.rand(1, 3, 480, 480) if type !='prn' else torch.rand(1, 56, 36, 17)
    with torch.no_grad():
        torch.onnx.export(model, input_tensor, export_path, export_params=True, opset_version=10)