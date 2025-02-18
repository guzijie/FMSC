"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def orthonorm_op1(Q,eps=1e-7):
#     m=torch.tensor(Q.shape[0],dtype=float)
#     outer_pord=torch.mm(Q.T,Q)
#     s=torch.eye(outer_pord.shape[0])
#     outer_pord=outer_pord.to('cpu')
#     outer_pord=outer_pord+eps*s
#
#     L=torch.cholesky(outer_pord) # 都不行，可以看出不是方法的错，就是这个拿到手的features（512,512）的行列式就是0的,0行太多了？？？ 不管是RuntimeError: cholesky_cpu: U(501,501) is zero, singular U.
#     L_inv=torch.inverse(L)
#     L_inv=L_inv.to('cuda')
#     return torch.sqrt(m) * L_inv.T

cantsolve = 0
def orthonorm_op1(x,epsilon=1e-7):
    global cantsolve
    # print("x.shape:",x.shape) # 都是feature后拿的，为什么一会儿torch.Size([250, 3, 32, 32])的，一会儿([250, 512])的
    x_2 = x.T @ x   # 是（512,200）叉乘（200,512维）那不应该是对512进行正交化 # x是{200,3,32,32} 转置的T就是{32,32,3,200}了， # x.transpose(2,3)是只转置了最后两维32*32的
    x_2 += torch.eye(x.shape[1],device=device)*epsilon # x_2是（512,512）的 ,这个512,512完全是对称的，但对称矩阵不一定行列式为0
    ## print("x_2:",x_2) # epsilon 太小了，e-7次方，根本前后一模一样，加在对角线上，torch.eye也是512,512的
    ## print("x_2.shape:",x_2.shape)
    # print("行列式=",torch.det(x_2).abs()) # 全部的行列式都为0了 行列式= tensor(0., device='cuda:0', grad_fn=<AbsBackward>)
    try:
        L = torch.cholesky(x_2) # colesky分解肯定是只能分解二维的矩阵 RuntimeError: cholesky_cuda: U(202,202) is zero, singular U.奇异矩阵:对应的行列式等于0,不可逆矩阵又叫奇异矩阵,因为绝大多数情况矩阵都是可逆的,也可以理解为它的行列式不为0。矩阵不可逆行列式一定为0
    except RuntimeError:
        # print("x_2:",x_2)
        # print("x_2.shape:",x_2.shape)
        cantsolve= cantsolve +1
        print("不能分解:",cantsolve)
        return x
    # temp=torch.sqrt(torch.tensor(x.shape[0],dtype=torch.float32)) # 那这里不应该只是根号200吗
    # temp1=torch.sqrt(torch.tensor(x_2.shape[0],dtype=torch.float32))
    ortho = L.inverse()*torch.sqrt(torch.tensor(x.shape[0],dtype=torch.float32)) # 乘上根号200 batch size
    ortho_weights = ortho.T
    # print('L.dtype:',L.dtype) # L.dtype: torch.float32 都是float32 ortho_weights..dtype: torch.float32
    out = x @ ortho_weights
    # print("out:",out)
    return out

# 不管是 expected cpu 还是 expected cuda， 本质原因都是类型不匹配。
# 一般是因为：
# 等号左边和右边类型不一样
# 运算符左右两端类型不同，例：+ - * /
# 同一个函数内，传入参数的类型不同，例matmul等

# torch.cholesky(input, upper=False, *, out=None) → Tensor
# RuntimeError: both arguments to matmul need to be at least 1D, but they are 0D and 0D
# b : tensor(0.2163, device='cuda:0', grad_fn=<SelectBackward>)
# b.shsape: torch.Size([])
# x.shape: torch.Size([200, 512])
# 为什么一会儿是（200,3,32,32）的，一会儿是（200,512）的，这512压成了一个维度肯定是做不了分解啊。难怪之前两层for取出来是0维的



def orthonorm_op(x,epsilon=1e-7):  # 是对一进来的x进行了正交化 # 应该对里面的每片32*32进行操作，包括进行cholesky分解的话，整个分解感觉不大对
    d=torch.zeros((3,3),dtype=float,device=device)
    e=torch.zeros((3,3),dtype=float,device=device)
    for i in range(x.shape[0]): # 200 # 跑了一段后 IndexError: too many indices for tensor of dimension 2
        for j in range(x.shape[1]): # 3
            b=x[i,j,] # 切出来的每一片都是32*32 拿进来的肯定是is_cuda bug在这一行IndexError: too many indices for tensor of dimension 2 b=x[i,j,:]最后的：删掉就可以了
            try:
                b_2=b.T @ b
            except RuntimeError:
                print("RuntimeError: both arguments to matmul need to be at least 1D, but they are 0D and 0D",b,"\nb.shsape:",b.shape,"\nx.shape:",x.shape)
                return x
            # epsilon=epsilon.to(device) # 但是这个好像没法todevice RuntimeError: expected device cuda:0 but got device cpu 加两句todevice
            epsilontensor=torch.eye(x.shape[3],device=device)*epsilon # epsilontensor.is_cuda True # 原来的shape[1]肯定不行 根据{200,3,32,32}写的是2或3维度
            # print("b_2.is_cuda",b_2.is_cuda) # b_2.is_cuda True
            # print("b_2",b_2)
            # print("torch.eye(x.shape[3])*epsilon",torch.eye(x.shape[3])*epsilon)
            # print("epsilontensor.is_cuda",epsilontensor.is_cuda) # 原来没有设置device时 epsilontensor.is_cuda False
            # print("epsilon.is_cuda",epsilon.is_cuda) # 'float' object has no attribute 'is_cuda'
            # print("torch.eye(x.shape[3])",torch.eye(x.shape[3]).is_cuda) # torch.eye(x.shape[3]) False
            b_2 =b_2 + epsilontensor

            det_b2 = torch.det(b_2).abs()
            try:
                L = torch.cholesky(b_2)
            except RuntimeError:
                det_b2=0 # 不看行列式了，直接看能不能分解，不能分的就跳过直接到else里去 x_new=b
                print("不能分解")
            if det_b2 > 0: # 行列式>0再可以求逆，分解矩阵
                print("det_b2",det_b2) # 行列式正的为什么还是不能分解 tensor(28066.4316, device='cuda:0')

                L = torch.cholesky(b_2) # cholesky_cuda: U(32,32) is zero, singular U.有些矩阵不可逆，没法求cholesky分解 矩阵 U 是上三角矩阵，如果upper为 False , 则返回的矩阵L为下三角
                ortho=L.inverse()*torch.sqrt(torch.tensor(x.shape[2],dtype=torch.float32))
                ortho_weights=ortho.T
                x_new = b @ ortho_weights
                # print("x_new:",x_new)
                # print("x_new:",x_new.shape)
            else:
                x_new=b # 等于原来的b，重新老样子拼接回去，不用正交了
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

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear': # Linear只有线性层，没有relu，mlp=linear+relu+linear稍微多一点
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(),
                    nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, withhead = True):
        if withhead:
          features = self.contrastive_head(self.backbone(x))
          features = F.normalize(features, dim = 1)
        else:
          features = self.backbone(x)
          features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):# 输入进来的 对比学习刚输出的特征？model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    def __init__(self, backbone, nclusters, nheads=1): # nclusters, nheads，num_classes: 10 num_heads: 3
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone'] # 对比学习的backbone和这里的backbone都是一样的
        self.backbone_dim = backbone['dim'] # 512? resnet18:return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}
        self.nheads = nheads  # num_heads: 3
        # 准备 在这里加正交层 加上 #######
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0) # 那这聚类头 是有3个头？？，3层线性层？
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        # self.temp_feature_orthonorm

    def forward(self, x, forward_pass='default'):    # 先backbone再head anchors_features = model(anchors, forward_pass='backbone')
        if forward_pass == 'default': # 没参数default的，都是写明了先backbone再head
            # x = orthonorm_op(x)  # 有个函数 对矩阵切片600张32*32右乘正交矩阵后再组合回（200,3,32,32）
            features = self.backbone(x)    # 是不是应该放在features后面啊，等会儿看下features大小多少，再调整，难道说这里的backbone就是对比学习调好了的参数？          # 输入为x,再通过resnet 18里拿出来 经过3个线性层  #这里的特征不应该是从对比学习里出来的吗？怎么只用了backbone
            # 加mlp
            features = orthonorm_op(features) # default是不是根本没进去过，都是要么backbone，要么head,要么return all
            out = [cluster_head(features) for cluster_head in self.cluster_head] # 有特征输入x扔进去的 anchors_output = model(anchors_features, forward_pass='head') # 只有head的，就只用线性层cluster_head
            print("globalcantsole:",cantsolve)

        elif forward_pass == 'backbone': # 只有backbone的，就只通过resnet backbone
            # x= orthonorm_op(x)
            out = self.backbone(x) # out是（200,512）
            # print("feature的形状：",out.shape)
            # out = orthonorm_op(out)

        elif forward_pass == 'head': # 只有head的，就只用线性层cluster_head # model(anchors_features, forward_pass='head')这时候传进来的x就是features
            # x= orthonorm_op(x) #第一次的模型是全都加了这句，所以是进backbone前也 # 在head前加效果和在backbone最后面加是一样的，都是没有一个能分解，而且都是在跑完600个以后，从512变成了3,32,32 真nm无语
            out = [cluster_head(x) for cluster_head in self.cluster_head] # OUT是{list:3},每个都是tensor(200,10)
            out = [orthonorm_op1(out[i]) for i in range(0,self.nheads)] # 最终用的的哪个聚类头的总loss小，就用哪个的，所以直接3个都正交化，总会用到，复杂度就多了两个20*20的矩阵或10*10的矩阵
            # out = orthonorm_op(out) # AttributeError: 'list' object has no attribute 'shape'
            # print("out:",out)
            # print("out(0)的形状：",out[0].shape) # out(0)的形状： torch.Size([250, 20]) ，cifar-20 有20个类
            # print("out(1)的形状：",out[1].shape) # out(1)的形状： torch.Size([250, 20])
            # print("out(2)的形状：",out[2].shape) # out(2)的形状： torch.Size([250, 20])
            # print("list len(out):",len(out)) #  len(out): 3 3个聚类头，应该是取最后一个吧，为什么要弄三个三个都是同样的shape但内容值不一样

        elif forward_pass == 'return_all':# 是最后eval 测试集的时候全部通过，所以训练的时候有正交层，测试的时候也要有正交层啊！服了
            # x= orthonorm_op(x)
            features = self.backbone(x)
            output=[cluster_head(features) for cluster_head in self.cluster_head]
            # out = {'features': features, 'output': output}
            output_ortho=[orthonorm_op1(output[i]) for i in range(0,self.nheads)]
            out = {'features': features, 'output': output_ortho}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
