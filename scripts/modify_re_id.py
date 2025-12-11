import os
import onnx
from onnx.tools import update_model_dims

model = onnx.load(R"C:\PureActiv\DL\Onnx\DefaultModels\re_id_orig.onnx")
# updated_model = update_model_dims.update_inputs_outputs_dims(model, {"inputs:0":[1,1024,2048,3]}, {"predictions:0":[1, 1025, 2049, 1]})
updated_model = update_model_dims.update_inputs_outputs_dims(model, {"input.1":[1,3,128,128]}, {"176":[1, 128]})
onnx.save(updated_model, R"C:\PureActiv\DL\Onnx\DefaultModels\re_id_orig-resize.onnx")


# python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch --dim_value 1 model.onnx model.fixed.onnx
# c:\PureActiv\DL\Onnx\DefaultModels\re_id_orig.onnx
# python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param input.1 --dim_value 1 re_id_orig.onnx re_id_orig-resize.onnx
# python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name input.1 --input_shape [1,3,128,128] re_id_orig.onnx re_id_orig-resize.onnx
