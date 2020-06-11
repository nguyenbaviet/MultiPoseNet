mo_path='/home/vietnguyen/intel/openvino/deployment_tools/model_optimizer/mo.py'
input_model='resnet101.onnx'
input_shape='[1,3,480,480]'
python ${mo_path} --input_model ${input_model} --input_shape ${input_shape} --log_level DEBUG --data_type FP32