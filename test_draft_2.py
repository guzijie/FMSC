import numpy as np
import torch
import torchvision
import torch.nn as nn
# x_2 = K.dot(K.transpose(x), x)  #应该是向量的叉乘，不是点乘（一个数值）
# x_2 += K.eye(K.int_shape(x)[1])*epsilon
# L = tf.cholesky(x_2)
# ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
# '张量矩阵的叉乘，也叫向量积，\na * b = |a| * |b| * cosθ， t1 @ t2 = ', t1 @ t2
# '张矩阵的点乘，也叫数量积，\na ∧ b = |a| * |b| * sinθ， t1 * t1 = ', t1 * t1
# [[ 0.4082483  -0.2809758   0.56954825]
#  [ 0.          0.5619516  -4.410687  ]
# [ 0.          0.          2.2649474 ]]

torch.set_printoptions(precision=6)
epsilon=1e-7
t1 = torch.tensor([[4.0,1.0 ,1.0],[1.0,2.0,3.0],[1.0,3.0,6.0]])
t2=t1.T @ t1
data1=t1.shape[0]
shape=torch.tensor(data1,dtype=torch.float32)# shape tensor(3.) #到底是要转成tensor:{3.0} 还是就一个数值 3.0 tf里是tensor的，但是tensor*tensor
alone=torch.eye(t1.shape[1])*epsilon
t2+=torch.eye(t1.shape[1])*epsilon
L=torch.cholesky(t2)
import numpy as np
A = np.array([[18.,  9., 13.],
              [ 9., 14., 25.],
              [13., 25., 46.]])

# 检查矩阵A是否对称正定
L = np.linalg.cholesky(A)
print("矩阵A是对称正定的，L：",L)
L=torch.from_numpy(L)
ortho_weights =(L.inverse()*torch.sqrt(shape)).T # sqrt(): argument 'input' (position 1) must be Tensor, not float
# print(shapet1.float()) # 'int' object has no attribute 'float'
#t2=torch.dot(t1.T,t1) # 1D tensors expected, got 2D, 2D tensors
print('t2',t2)
print('data1',data1) # 3
print('shape',shape)
print("sqrt",torch.sqrt(shape))
print(alone)
print('l:',L)
print('l.inverse:',L.inverse())
print('ortho_weights:',ortho_weights)
# l: tensor([[4.2426, 0.0000, 0.0000],
#            [2.1213, 3.0822, 0.0000],
#            [3.0641, 6.0022, 0.7647]])
# l: tensor([[4.24264050, 0.00000000, 0.00000000],
#            [2.12132049, 3.08220696, 0.00000000],
#            [3.06412959, 6.00219250, 0.76471949]])#设置显示8位精度
x=torch.tensor([[4.2426405 , 0.,  0. ],
                [2.1213205 , 3.082207 , 0. ],
               [3.0641296 , 6.0021925 , 0.76472014]],dtype=float)
print("tf的矩阵x.inverse()",x.inverse()) # 直接拿tf那边的L求逆答案是对的 所以说应该是精度的问题 ，换成8位精度的这里就大差不差了
# x.inverse() tensor([[ 0.2357,  0.0000, -0 .0000],
#                     [-0.1622,  0.3244,  0.0000],
#                     [ 0.3288, -2.5465,  1.3077]], dtype=torch.float64)
print("tf的ortho",(x.inverse()*torch.sqrt(shape)).T)# 直接拿tf那边的L求逆+乘+转置 答案是对的

t0 = torch.tensor([[4.0,1.0 ,1.0],[1.0,2.0,3.0],[1.0,3.0,6.0]]) # 这个确实是对称的矩阵
t1 = torch.tensor([[1.0,2.0,-1.0],[2.0,3.0,4.0],[3.0,1.0,2.0]])
#unrelugulartensor=torch.tensor([[1,2,3],[]])
print("test inverse:",t1.inverse())
#t1 = torch.tensor([[2.0,1.0 ,1.0,0.0],[4.0,3.0,3.0,1.0],[8.0,7.0,9.0,5.0],[6.0,7.0,9.0,8.0]])
def orthonorm_op(x,epsilon=1e-7):
    x_2=x.T @ x
    x_2+=torch.eye(x.shape[1])*epsilon
    L=torch.cholesky(x_2)
    ortho=L.inverse()*torch.sqrt(torch.tensor(x.shape[0],dtype=torch.float32))
    ortho_weights=ortho.T
    print('L.dtype:',L.dtype) # L.dtype: torch.float32 都是float32 ortho_weights..dtype: torch.float32
    return ortho_weights
#def Orthonorm(x, name=None):
d=x.size()
print("d:",d)
print("def print:",orthonorm_op(t1))
print("def print t0:",orthonorm_op(t0))
# def print: tensor([[ 4.08248305e-01, -2.80975789e-01,  5.69548726e-01],
#                    [ 0.00000000e+00,  5.61951458e-01, -4.41069031e+00],
#                    [-0.00000000e+00,  2.58095678e-08,  2.26494908e+00]])

# l=torch.tensor([[4.24264050, 0.00000000, 0.00000000],
#                 [2.12132049, 3.08220696, 0.00000000],
#                 [3.06412959, 6.00219250, 0.76471949]],dtype=float)
# print("直接L inverse:",L.inverse())
#
# x=np.array([[4.24264050, 0.00000000, 0.00000000],
#             [2.12132049, 3.08220696, 0.00000000],
#             [3.06412959, 6.00219250, 0.76471949]])
# print("np里np.linalg.inv",np.linalg.inv(l))
# # 如果说 对pytorch这里的L，3钟环境分解都不一样，那会不会是cholesky分解得到的L有问题，试试看如果对tf里的L进行逆矩阵运算
#
# import numpy as np
# A = np.array([[18.,  9., 13.],
#               [ 9., 14., 25.],
#               [13., 25., 46.]])
# # 检查矩阵A是否对称正定
# try:
#     L = np.linalg.cholesky(A)
#     print("矩阵A是对称正定的，L：",L)
# except np.linalg.LinAlgError:
#     print("矩阵A不是对称正定的")
l=torch.tensor([[[4.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]],
                [[4.24264050, 0.00000000, 0.00000000],
                 [2.12132049, 3.08220696, 0.00000000],
                 [3.06412959, 6.00219250, 0.76471949]]],dtype=float)
print("l.transpose(1,2):",l.transpose(1,2))
A = torch.tensor([[18.,  9., 13.],
                    [ 9., 14., 25.],
                    [13., 25., 46.]],dtype=float)
print("行列式A=",torch.det(A).abs())
# 行列式A= tensor(100.000000, dtype=torch.float64) 行列式不为0 可以分解cholesky 可以求逆矩阵

def orthonorm_op1(Q,eps=1e-7):
    m=torch.tensor(Q.shape[0],dtype=float)
    outer_pord=torch.mm(Q.T,Q)
    s=torch.eye(outer_pord.shape[0])
    outer_pord=outer_pord.to('cpu')
    outer_pord=outer_pord+eps*s

    L=torch.cholesky(outer_pord)
    L_inv=torch.inverse(L)
    L_inv=L_inv.to('cuda')
    return torch.sqrt(m) * L_inv.T

print("qing_orthonorm_op1(t0):",orthonorm_op1(t0))
print("qing_orthonorm_op1(t1):",orthonorm_op1(t1))