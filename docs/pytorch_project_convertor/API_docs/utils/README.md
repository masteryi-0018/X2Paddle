## 工具类API映射列表
该文档梳理了与数据处理、分布式处理等相关的PyTorch-PaddlePaddle API映射列表。
| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=dataparallel#torch.nn.DataParallel) | [paddle.DataParallel](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dygraph/parallel/DataParallel_cn.html#dataparallel) | [差异对比](torch.nn.DataParallel.md)                   |
| 2    | [torch.nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=torch%20nn%20parameter#torch.nn.parameter.Parameter) | [paddle.create_parameter](https://github.com/PaddlePaddle/Paddle/blob/ce2bdb0afdc2a09a127e8d9aa394c8b00a877364/python/paddle/fluid/layers/tensor.py#L77) | [差异对比](torch.nn.parameter.Parameter.md)            |
| 3    | [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_) | 无对应实现                                                   | [组合实现](torch.nn.utils.clip_grad_value_.md)         |
| 4    | [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) | [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/reader/DataLoader_cn.html#dataloader) | [差异对比](torch.utils.data.DataLoader.md)             |
| 5    | [torch.utils.data.random_split](https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split) | 无对应实现                                                   | [组合实现](torch.utils.data.random_split.md)           |
| 6    | [torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler) | 无对应实现                                                   | [组合实现](torch.utils.data.distributed.DistributedSampler.md) |
| 7    | [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.Dataset) | [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/dataset/Dataset_cn.html#dataset) | 功能一致                                                     |
| 8    | [torch.utils.data.BatchSampler](https://pytorch.org/docs/stable/data.html?highlight=batchsampler#torch.utils.data.BatchSampler) | [paddle.io.BatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/batch_sampler/BatchSampler_cn.html#batchsampler) | [差异对比](torch.utils.data.BatchSampler.md)           |
| 9    | [torch.utils.data.Sampler](https://pytorch.org/docs/stable/data.html?highlight=sampler#torch.utils.data.Sampler) | [paddle.io.Sampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/sampler/Sampler_cn.html#sampler) | 功能一致                                                     |

***持续更新...***