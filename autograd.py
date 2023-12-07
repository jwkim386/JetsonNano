import torch
from torch.autograd import Variable

# x = torch.ones(2,2)

# x = Variable(x, requires_grad=True) #사용자 정의 변수는 필히 requires_grad=True 선언해줘야함. CNN, RNN 모델에서는 자동선언되어 있음

# print(x)
# print(x.data) #tensor형태의 data
# print(x.grad) #data가 거쳐온 layer에 대한 미분값 축적
# print(x.grad_fn)#미분값을 계산한 함수에 대한 정보

# y = x + 2
# print(y)

# z = y**2
# print(z)

# out = z.sum()
# print(out)
# out.backward()

# print(x.data)
# print(x.grad)
# print(x.grad_fn)

# print(y.data)
# print(y.grad)
# print(y.grad_fn)

# print(z.data)
# print(z.grad)
# print(z.grad_fn)

# print(out.data)
# print(out.grad)
# print(out.grad_fn)

x = torch.ones(3)
x = Variable(x, requires_grad=True)
y = x**2
z = y*3
print(z)
grad = torch.Tensor([0.1, 1, 10])
z.backward(grad)

print('----x.data----')
print(x.data)
print('----x.grad----')
print(x.grad)
print('----x.grad_fn----')
print(x.grad_fn)
