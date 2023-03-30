import torch
import torch.nn as nn
import torch.nn.functional as F
hc=torch.randn(10,3)
indice=torch.tensor([[2,3,3],[2,4,8]])
value=torch.tensor([1.,1.,1.])
vadj=torch.sparse_coo_tensor(indices=indice,values=value,size=[5,20])
hv=torch.randn(5,3)

h=torch.spmm(hv,hc.transpose(0,1))

out=torch.cat([F.softmax(h),F.softmax(h)],dim=1)
att=torch.mul(vadj.to_dense(),out)
print(att)
print(att.shape)