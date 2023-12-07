# import torch 
# from torch.autograd import Variable 
# import torch.nn as nn 
# import numpy as np

# input = torch.ones(1,1,3,3)
# input = Variable(input, requires_grad=True)
# filter = nn.Conv2d(1,1,3) #Conv2d 모델 사용
# print(filter.weight)
# out = filter(input)
# print(out)
# print(input.grad)
# out.backward()
# print(out.grad_fn)
# print(input.grad)
# print(input.grad.sum())

import torch 
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F

input = torch.ones(1,1,5,5)
input = Variable(input, requires_grad=True)
filter = nn.Conv2d(1,1,3,bias=None)
print(filter.weight)
filter.weight = nn.Parameter(torch.ones(1,1,3,3) + 1)
filter.weight
out = filter(input)
print(out)
