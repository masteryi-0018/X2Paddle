import paddle
import numpy as np
import sys
import pickle

f = open('result.txt', 'w')
f.write("======SegFlow: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CUDAPlace(0))

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    with open("../dataset/SegFlow/inputs_segflow_0314.pkl", "rb") as fr:
        input_list = pickle.load(fr)
    result = exe.run(prog,
                     feed={
                         inputs[0]: input_list[0],
                         inputs[1]: input_list[1]
                     },
                     fetch_list=outputs)

    with open("../dataset/SegFlow/output_segflow_0314.pkl", "rb") as fr:
        caffe_result = pickle.load(fr)

    is_successd = True
    for i in range(2):
        diff = result[i] - caffe_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff_all = max_abs_diff / np.fabs(caffe_result[i]).max()
            relative_diff = relative_diff_all.max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
