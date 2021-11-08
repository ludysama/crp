from numpy import degrees, printoptions
import torch
import torch.nn.functional as F
# ls = [-1,0,1,2,3,4]
# ls = [-222,1,2,3,4,6]
# ls = torch.Tensor(ls)
# print(F.normalize(ls,p=0,dim=0))
# print(F.normalize(ls,p=1,dim=0))
# print(F.normalize(ls,p=2,dim=0))
# print(F.normalize(ls,dim=0))

# queue_size = 16
# ridx = torch.zeros(2, queue_size, dtype=torch.long)
# print(ridx,ridx.shape)
# rot_label1 = torch.Tensor([1,2,3,0])
# rot_label2 = torch.Tensor([1,3,3,0])
# rot_lbs = torch.stack((rot_label1, rot_label2))
# print(rot_lbs, rot_lbs.shape)

# rot_label_0 = torch.LongTensor([i for k in range(5) for i in range(4) for j in range(3)])
# print(rot_label_0,rot_label_0.shape)
# rot_label_0 = torch.stack([rot_label_0]*2)
# print(rot_label_0,rot_label_0.shape)


# rot_label = torch.LongTensor([i for i in range(4) for k in range(8 // 4)])
# print(rot_label)
# mask = torch.randperm(8)
# print(mask)
# rot_label = rot_label[mask]
# print(rot_label)
# rot_label_template = torch.LongTensor([i for i in range(4) for k in range(16 // 4)])
# print(rot_label_template)
# mask2 = torch.randperm(16)
# rot_label_template = rot_label_template[mask2]
# print(rot_label_template)
# mask3 = rot_label.T.unsqueeze(dim=1) == rot_label_template
# print(mask3.shape)
# print(mask3)
# a = torch.randn(mask3.shape)
# print(a)
# print(a[mask3].view(a.shape[0],-1))

# rot_label1 = torch.LongTensor([0,1,2,3,0,1,2,3])[torch.randperm(8)]
# rot_label2 = torch.LongTensor([0,1,2,3,0,1,2,3])[torch.randperm(8)]
# relative_label = (rot_label2 - rot_label1)%4
# print(rot_label1)
# print(rot_label2)
# print(relative_label)

# def rotated_loc_code(loccode:int, degree:int, split:int, mode:str = 'normal')->int:
#     '''
#         if split == 2:
#         then the code is 0 1            if degree == 1 then 1 3
#                             2 3                                0 2
#         if split == 3:
#         then the code is 0 1 2
#                             3 4 5
#                             6 7 8
#         if split == 4:
#         then the code is 0  1  2  3     if degree == 1 then 3  7  11 15
#                             4  5  6  7                         2  6  10 14
#                             8  9  10 11                        1  5  9  13
#                             12 13 14 15                        0  4  8  12
#     '''
#     if mode == 'counter':
#         degree = (4 - degree) % 4

#     for i in range(degree):
#         code_row = loccode // split
#         code_collumn = loccode % split
#         code_row_new = code_collumn
#         code_collumn_new = split - code_row -1
#         loccode = code_row_new * split + code_collumn_new
        
#     return loccode
# dense_split = 3
# template = torch.LongTensor([[i for i in range(dense_split**2)] for j in range(4)])
# print(template)

# for i in range(1,4):
#     for j in range(dense_split**2):
#         template[i][j] = rotated_loc_code(j,i,dense_split)
# print(template)

# res = torch.LongTensor([0,1,2,3,0,1,2,3])[torch.randperm(8)]
# x = torch.randn((8,9,3))
# print(x)
# print(res)
# # x = x.permute(0,2,1)
# # x = x[:,[2,5,8,1,4,7,0,3,6],:]
# for i in range(4):
#     x[res==i] = x[res==i][:,template[i],:]
# print(x)

# dense_label1 = torch.stack([res.clone() for j in range(3**2)], dim=0)
# print(dense_label1)

# x = torch.randn((2,3),requires_grad=True)
# y = torch.randn((2,3),requires_grad=True)
# print(x)
# print(y)
# xcaty = torch.cat((x,y),dim=1)
# ycatx = torch.cat((y,x),dim=1)
# print(xcaty)
# print(ycatx)
# xcaty= 2.1 * xcaty
# print(x)
# print(y)
# print(xcaty)
# print(ycatx)

# print(xcaty.grad_fn)
# print(ycatx.grad_fn)

# print(xcaty.detach().grad_fn)
# print(ycatx.detach().grad_fn)

# print(xcaty.grad_fn)
# print(ycatx.grad_fn)

x = torch.randn((2,4))
y = torch.LongTensor([[0,1,2],[1,2,3]])
print(x)
print(y)
yadd = torch.LongTensor([[i*y.shape[1] for j in range(y.shape[1])] for i in range(y.shape[0])])
print(yadd)

y = y + yadd
print(y)
xsel = torch.take(x,y)
print(xsel)