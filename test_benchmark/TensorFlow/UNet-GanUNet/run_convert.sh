# 进行转换
x2paddle -f tensorflow -m ../dataset/UNet-GanUNet/gan_unet.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
