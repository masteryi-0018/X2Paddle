# 进行转换
x2paddle -f onnx -m ../dataset/vgg19_bn/model.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py