# 进行转换
x2paddle -f tensorflow -m ../dataset/VGG16/vgg16.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py