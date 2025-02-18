import numpy as np
import torch
import torchvision
import torch.nn as nn
a=torch.tensor([[
                [[1.24264050, 0.00000000, 0.00000000],
                    [2.12132049, 3.08220696, 0.00000000],
                    [3.06412959, 6.00219250, 0.76471949]],
                   [[2.24264050, 0.00000000, 0.00000000],
                    [2.12132049, 3.08220696, 0.00000000],
                    [3.06412959, 6.00219250, 0.76471949]]],
                [[[3.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]],
                [[4.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]]],
                [[[5.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]],
                [[6.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]]]],dtype=float)
print("a.shape:",a.shape) # torch.Size([3, 2, 3, 3])  有6片3*3
z=torch.zeros((3,3),dtype=float)
e=torch.zeros((3,3),dtype=float)
e=torch.unsqueeze(e,0)
for i in range(a.shape[0]): # 200
    for j in range(a.shape[1]): # 3
        b=a[i,j,:]
        if j<(a.shape[1]-1):
            c=a[i,j+1,:]
            z=torch.stack((b,c),0)
            print("z=stack((b,c),0)",z)
    # e=torch.stack((e,z),0)  # 是旧的tensor stack上叠新的tensor， 但是stack需要两张量大小一致，一大一小没法stack

print("e.shape",e.shape)
print("e",e)
print("z.shape",z.shape) # z.shape torch.Size([21, 3])
c=a[0,0,:]
d=torch.stack((b,c),0) # 用stack((b,c),0)是可以的，用cat((b,c),0)直接就变成了[6,3]
print("d:",d)
print("d.shape:",d.shape) # torch.Size([2, 3, 3])
e=torch.cat((b,c),0)
print("e:",e)
print("e.shape:",e.shape)

# 在pytorch,鉴于张量a的形状(1X11)和 b造型(1X11),torch.stack((a,b),0)都会给我形状的张量(2X11)
#
# 但是,当a形状(2X11)和b形状时(1X11),torch.stack((a,b),0)会引起错误cf. "两个张量大小必须完全相同".
#
# 因为两个张量是模型的输出(包括渐变),我无法将它们转换为numpy使用np.stack()或np.vstack().
#
# 是否有任何可能的解决方案,至少GPU内存使用？
#
# 您似乎想要使用torch.cat()(沿现有维度连接张量)而不是torch.stack()(沿新维度连接/堆栈张量):
#
# import torch
# a = torch.randn(1, 42, 1, 1)
# b = torch.randn(1, 42, 1, 1)
#
# ab = torch.stack((a, b), 0)
# print(ab.shape)
# # torch.Size([2, 1, 42, 1, 1])
#
# ab = torch.cat((a, b), 0)
# print(ab.shape)
# # torch.Size([2, 42, 1, 1])
# aab = torch.cat((a, ab), 0)
# print(aab.shape)
# # torch.Size([3, 42, 1, 1])

# if j<(a.shape[1]-1):
#     c=a[i,j+1,:]
#     c=torch.unsqueeze(c,0)
#     d=torch.cat((d,c),0)
a=torch.tensor([[
    [[1.24264050, 0.00000000, 0.00000000],
     [2.12132049, 3.08220696, 0.00000000],
     [3.06412959, 6.00219250, 0.76471949]],
    [[2.24264050, 0.00000000, 0.00000000],
     [2.12132049, 3.08220696, 0.00000000],
     [3.06412959, 6.00219250, 0.76471949]],
    [[3.24264050, 0.00000000, 0.00000000],
     [2.12132049, 3.08220696, 0.00000000],
     [3.06412959, 6.00219250, 0.76471949]]],
    [[[4.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[5.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[6.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]]],
    [[[7.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[8.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[9.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]]],
    [[[10.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[11.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]],
     [[12.24264050, 0.00000000, 0.00000000],
      [2.12132049, 3.08220696, 0.00000000],
      [3.06412959, 6.00219250, 0.76471949]]]],dtype=float) # torch.Size([4, 3, 3, 3]) 12片3*3
d=torch.zeros((3,3),dtype=float)
e=torch.zeros((3,3),dtype=float)
for i in range(a.shape[0]): # 200
    for j in range(a.shape[1]): # 3
        b=a[i,j,:]
        b=torch.unsqueeze(b,0)
        if j==0:
            d=b
        else:
            d=torch.cat((d,b),0)
    print("每一大片d：",d)
    print("每一大片d：",d.shape)
    d=torch.unsqueeze(d,0)
    if i==0:
        e=d
    else:
        e=torch.cat((e,d),0)
print("最终大片e：",e)
print("最终大片e：",e.shape)  # 最终大片e： torch.Size([4, 3, 3, 3])

def orthonorm_op(x,epsilon=1e-7):
    d=torch.zeros((3,3),dtype=float)
    e=torch.zeros((3,3),dtype=float)
    for i in range(x.shape[0]): # 200
        for j in range(x.shape[1]): # 3
            b=x[i,j,:] # 切出来的每一片都是32*32

            b_2=b.T @ b
            b_2+=torch.eye(b.shape[1])*epsilon
            print("b.shape[1]:",b.shape[1])
            L=torch.cholesky(b_2)
            ortho=L.inverse()*torch.sqrt(torch.tensor(b.shape[0],dtype=torch.float32))
            ortho_weights=ortho.T
            x_new = b @ ortho_weights
            print("x_new:",x_new)
            print("x_new:",x_new.shape)

            b=torch.unsqueeze(x_new,0) # 这是对b进行重新拼接回来，最后应该是对b右乘完权重矩阵后再拼接回来。
            if j==0:
                d=b
            else:
                d=torch.cat((d,b),0)
        # print("每一大片d：",d)
        # print("每一大片d：",d.shape)
        d=torch.unsqueeze(d,0)
        if i==0:
            e=d
        else:
            e=torch.cat((e,d),0)

    return e

e=orthonorm_op(a)
print("最终大片：",e)
print("最终大片：",e.shape) # 最终大片： torch.Size([4, 3, 3, 3])

matrices = torch.randn([5,3,3]) # 5片3*3
print("matrices:",matrices)
matrices[[2,3]] = torch.zeros([3,3]) # 中间两片换成全0了
print("matrices[[2,3]]",matrices)
determinants = torch.det(matrices) # 这5片3*3的行列式 tensor([ 0.8127, -1.5897,  0.0000,  0.0000, -0.5652])
print("determinants",determinants)

try:
    results_tensor=torch.cholesky(torch.zeros([3,3]))
except RuntimeError:
    print("RuntimeError: cholesky_cuda: U(32,32) is zero, singular U.")
# 只输出了这 RuntimeError: cholesky_cuda: U(32,32) is zero, singular U.
inverses = torch.inverse(matrices[determinants.abs()>0.]) # 拿大于0的去做
print("inverses",inverses)

# 看看 a矩阵里有几个行列式>0的
det_a=torch.det(a) # a是[4, 3, 3, 3]
print("det_a",det_a) # 所以det应该是[4,3] 居然这12片全是正的，全部可以cholesky分解
# det_a tensor([[ 2.9289,  5.2860,  7.6430],
#               [10.0000, 12.3570, 14.7141],
#               [17.0711, 19.4281, 21.7851],
#               [24.1421, 26.4992, 28.8562]], dtype=torch.float64)

anew=a.reshape(12,3,3)
split1=torch.split(anew,1,0) # 按间隔为1，在第0 个维度堵上每隔1取
# print("split1",split1)
for i in split1:
    print("i in split1",i)
    print("i in split1",i.shape) # ([1, 3, 3, 3]) 不会降低维度的，只是把4变成了1
    i=torch.squeeze(i,0)
    print("squeeze_i",i)
    print("squeeze_i in split1",i.shape) # squeeze_i in split1 torch.Size([3, 3, 3])
# print("直接squeeze：",torch.squeeze(split1,0)) # didn't match , invalid types: (tuple, int)
for i in split1:
    print("squeeze_i in split1",i.shape) # squeeze_i in split1 torch.Size([1, 3, 3, 3]) tuple里不会变，那就自己再里面再改好了

# 如果不用 for改用 tuple也会这样吗？出现IndexError: too many indices for tensor of dimension 2吗
# b=x[i,j,:] # 切出来的每一片都是32*32 拿进来的肯定是is_cuda
# IndexError: too many indices for tensor of dimension 2
# bug应该是说0,1,2，第2维度的索引太多了，那换种写法？或者换成split?
print("ceshi取切片：",a[1,1,])

# RuntimeError: both arguments to matmul need to be at least 1D, but they are 0D and 0D
# b : tensor(0.2163, device='cuda:0', grad_fn=<SelectBackward>)
# b.shsape: torch.Size([])
# x.shape: torch.Size([200, 512])
# 为什么一会儿是（200,3,32,32）的，一会儿是（200,512）的，这512压成了一个维度肯定是做不了分解啊。难怪之前两层for取出来是0维的

lengtha = a.shape



