from __future__ import print_function
import paddle
import sys
import os
import numpy as np
import pickle

f = open("result.txt", "w")
f.write("======VGG19: \n")

try:
    with open('../dataset/VGG19/caffe_input.pkl', 'rb') as inp:
        input_data = pickle.load(inp)["data0"]
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygrah
    [prog, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(
         path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    result = exe.run(prog,
                     feed={feed_target_names[0]: input_data},
                     fetch_list=fetch_targets)
    f.write("Dygraph Successed\n")
except:
    f.write("!!!!!Failed\n")
f.close()
