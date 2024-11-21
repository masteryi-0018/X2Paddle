import paddle
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======pnasnet_mobile: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    data = np.load('../dataset/pnasnet_mobile/input.npy')
    # data = np.random.rand(2,224,224,3).astype("float32")
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    with open("../dataset/pnasnet_mobile/output.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)
    is_successd = True
    for i in range(1):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff = max_abs_diff / np.fabs(onnx_result[i]).max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")