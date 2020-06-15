mo_path='/home/vietnguyen/intel/openvino/deployment_tools/model_optimizer/mo.py'
input_model='/home/vietnguyen/refactor/demo/convert/mobilenetv2_detectANDkps.onnx'
#input_shape='[1,56,36,17]' #prn
input_shape='[1,3,480,480]' #detectionANDkps
output_dir='/home/vietnguyen/refactor/demo/convert/openvino/'
python ${mo_path} --input_model ${input_model} --input_shape ${input_shape} --output_dir ${output_dir} --data_type FP32