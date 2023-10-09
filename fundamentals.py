import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())

print(torch.__version__)

## tensors:

# creating:
# scalar:
scalar = torch.tensor(7)
print(scalar)

print(scalar.ndim)

print(scalar.item())

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

MATRIX = torch.tensor([[7,8], 
                       [9,10]])
print(MATRIX)
print(MATRIX[1])
print(MATRIX.ndim)
print(MATRIX.shape)

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

# random tensors
random_tensor = torch.rand(3, 4) # randdom tensor of size (3, 4)
print(random_tensor) 
print(random_tensor.ndim) 

# create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3, 224, 224)) # color channels, heigh, width
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# zeros and ones
zeros = torch.zeros(size=(3,4))
print(zeros)

ones = torch.ones(3,4)
print(ones)

# range of tensors / tensors like
arange = torch.arange(0, 10)
print(arange)

arange2 = torch.arange(start=0, end=30, step=3)
print(arange2)

ten_zeros = torch.zeros_like(input=arange)
print(ten_zeros)

# tensors datatypes
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # datatype
                               device=None, # "cpu" or "cuda"
                               requires_grad=False)
print(float_32_tensor)
print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

print(float_32_tensor * float_16_tensor)

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(float_32_tensor * int_32_tensor)

# tensors operations + * - / and matrix multiplication

tensor = torch.tensor([1,2,3])
print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
print(tensor / 10)

print(torch.mul(tensor, 10))


## matrix multiplication
# 2 ways: 
# element-wise 
print(tensor)
print(tensor * tensor)

# and by matrix
print(torch.matmul(tensor, tensor))
print(tensor @ tensor)

# transpose
tensor = torch.rand(2,3) 
print(tensor, tensor.shape)
print(tensor.T, tensor.T.shape)
print(torch.matmul(tensor,tensor.T))

## aggregation
print(tensor.min())
print(tensor.max())
print(tensor.mean())
print(tensor.sum())

# position of max and min
print(tensor.argmin())
print(tensor.argmax())

## Reshaping, View, Staking, Squeezing, Unsqueezing, Permute

x = torch.arange(1., 11.)
print(x, x.shape)

# add dimension
x_reshaped = x.reshape(5, 2)
print(x_reshaped, x_reshaped.shape)

# change the view (shares the same memory)
z = x.view(5, 2)
z[:, 0] = 5
print(z)
print(x)

# stack
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)

# squeeze
x_reshaped = x.reshape(1, 10)
print(x_reshaped, x_reshaped.shape)
print(x_reshaped.squeeze(), x_reshaped.squeeze().shape)

# unsqueeze
x_unsqueeze = x_reshaped.unsqueeze(dim=0)
print(x_unsqueeze, x_unsqueeze.shape)
x_unsqueeze = x_reshaped.unsqueeze(dim=2)
print(x_unsqueeze, x_unsqueeze.shape)

# permute (shares the memory)
x_ori = torch.rand(size=(224, 100, 3)) # iamge H, W, color channels
print(x_ori.shape)
x_permuted = x_ori.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(x_permuted.shape)

## indexing (similar to numpy)
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print(x[0])
print(x[0][0])
print(x[0][0][0])
# can use : to select all of the dimension
print(x[:, :, 1])

# numpy <-> pytorch
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

array = array + 1
print(array)
print(tensor) # they do not share memory 


tensor = torch.ones(7)
np_tensor = tensor.numpy()
print(tensor, np_tensor)

tensor = tensor + 1
print(tensor) 
print(np_tensor) # they do not share memory 

# random seed
randA = torch.rand(3,4)
randB = torch.rand(3,4)
print(randA == randB)

# set random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
randC = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
randD = torch.rand(3,4)

print(randC == randD)

# runnning on GPU
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())

tensor = torch.tensor([1, 2, 3])
print(tensor.device)

tensor_gpu = tensor.to(device)
print(tensor_gpu, tensor_gpu.device)

# if tensor on gpu, can't transform into numpy
tensor_np = tensor_gpu.cpu().numpy()
print(tensor_np)









