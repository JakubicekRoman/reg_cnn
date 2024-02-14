import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage, random_noise
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage import io, color, transform
from skimage.transform import SimilarityTransform, warp, rotate
from torchvision.transforms import GaussianBlur

torch.manual_seed(0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        nK = 2
        dilation = 1
        sk = 2
        self.conv1 = nn.Conv2d(in_channels+1, out_channels*nK, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels*nK, out_channels*nK, kernel_size, stride=stride*sk, padding=padding, dilation=dilation)
        self.conv3 = nn.Conv2d(out_channels*nK, out_channels*nK, kernel_size, stride=stride*sk, padding=padding, dilation=dilation)
        self.conv4 = nn.Conv2d(out_channels*nK, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.kernelBlur = GaussianBlur((7,7),(2,2))
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        f1 = self.conv1(x)
        f = self.conv2(f1)
        f = self.relu(f)
        f = self.conv3(f)
        # f = self.conv3(f+f1)
        f = F.interpolate(f,size=(x.size(2),x.size(3)))
        f = self.conv4(f + f1)
        # f = self.kernelBlur(f)+x[:,0,:,:]
        x = self.relu(f) + x[:,0,:,:]
        # x = self.relu(f) + x[:,0,:,:] + x[:,1,:,:]
        # x = self.relu(f)
        # x = self.kernelBlur(x)
        # f = self.tanh(f)

        return x
    
class SimpleModel(nn.Module):
    def __init__(self, in_channels, out_channels, weights):
        super(SimpleModel, self).__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block_2 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block_3 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block_4 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block_5 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_reg = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=(1,1), bias=False)
        self.conv_reg.weight = nn.Parameter(weights)
        self.conv_reg.weight.requires_grad = False
        self.kernelBlur = GaussianBlur((11,11),(3,3))

    def forward(self, x, ref):
        output_Reg_1 = self.conv_reg(x)
        nb = output_Reg_1.shape[1]
        feat = self.conv_block_1(torch.cat((x,x-ref),1))
        att = F.softmax(feat, dim=1)
        att = self.kernelBlur(att)
        x = torch.sum(att * output_Reg_1, 1).view(1,1,att.size(2),att.size(3))
        attAll = att.clone()

        output_Reg_1 = self.conv_reg(x)
        nb = output_Reg_1.shape[1]
        feat = self.conv_block_2(torch.cat((x,x-ref),1))
        att = F.softmax(feat, dim=1)
        att = self.kernelBlur(att)
        x = torch.sum(att * output_Reg_1, 1).view(1,1,att.size(2),att.size(3))
        attAll = attAll+att

        output_Reg_1 = self.conv_reg(x)
        nb = output_Reg_1.shape[1]
        feat = self.conv_block_3(torch.cat((x,x-ref),1))
        # att = F.softmax(feat, dim=1)
        x = torch.sum(att * output_Reg_1, 1).view(1,1,att.size(2),att.size(3))
        attAll = attAll+att

        output_Reg_1 = self.conv_reg(x)
        nb = output_Reg_1.shape[1]
        feat = self.conv_block_4(torch.cat((x,x-ref),1))
        att = F.softmax(feat, dim=1)
        # att = self.kernelBlur(att)
        x = torch.sum(att * output_Reg_1, 1).view(1,1,att.size(2),att.size(3))
        attAll = attAll+att

        output_Reg_1 = self.conv_reg(x)
        nb = output_Reg_1.shape[1]
        feat = self.conv_block_5(torch.cat((x,x-ref),1))
        att = F.softmax(feat, dim=1)
        # att = self.kernelBlur(att)
        x = torch.sum(att * output_Reg_1, 1).view(1,1,att.size(2),att.size(3))
        attAll = attAll+att

        # kernelBlur = GaussianBlur((21,21),(5,5))
        # att = kernelBlur(att)

        att1 = torch.argmax(att, dim=1)[0,:]
        att1 = F.one_hot(att1, num_classes=nb)
        att1 = att1.permute(2,0,1).view(1,att1.size(2),att1.size(0),att1.size(1))
        output = torch.sum(att1*output_Reg_1,1)

        return x, output, att, attAll

nb_channels = 1
# h, w = 15, 15
# x = torch.randn(1, nb_channels, h, w)

# image_path = "D:\Projekty\Registration_my\stock.jpg"
image_path = "D:\Projekty\Registration_my\\bf.jpg"
x = io.imread(image_path)
x = color.rgb2gray(x)
x = transform.rescale(x,0.20,order=0)
# x = transform.rescale(x,0.015,order=0)
# x = x[0:-3,:]

# ref = x > 0.6
# x = x > 0.6

# x = random_noise(x, mode='gaussian')
ref = copy.deepcopy(x)


# tform = SimilarityTransform(translation=(2, 0))
# x[0:int(x.shape[0]/2),:] = warp(x[0:int(x.shape[0]/2),:], tform)
# tform = SimilarityTransform(translation=(-3, 0))
# x[int(x.shape[0]/2):-1,:] = warp(x[int(x.shape[0]/2):-1,:], tform)

# tform = SimilarityTransform(translation=(-3, 2))
# x = warp(x, tform, order=0)

tform = SimilarityTransform(translation=None, scale=0.98)
x = warp(x, tform, order=0)

# x = rotate(x, 2, resize=False, order=0)

plt.figure()
plt.imshow(x)
plt.show() 

plt.figure()
plt.imshow(ref)
plt.show()

nr, nc = x.shape
im = np.zeros((nr, nc, 3))
im[..., 0] = ref
im[..., 1] = x
im[..., 2] = x
plt.figure()
plt.imshow(im)

x = torch.from_numpy(x).view(1,1,x.shape[0],x.shape[1]).float().to("cuda")
ref = torch.from_numpy(ref).view(1,1,ref.shape[0],ref.shape[1]).float().to("cuda")

# x = Variable(x,  requires_grad=True)

nb_kernels = 9
mat = np.zeros([nb_kernels,3,3])
vec = [1.,0.,0.,0.,0.,0.,0.,0.,0.]
mat[0,::] = np.reshape(vec,[3,3])
for i in range(1,9,1):
    vec = np.roll(vec,1)
    mat[i,::] = np.reshape(vec,[3,3])
weights = torch.tensor(mat, dtype=torch.float32)
weights = weights.view(nb_kernels, nb_channels, 3, 3)

model = SimpleModel(in_channels=1, out_channels=9, weights=weights).to("cuda")
criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
# non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(non_frozen_parameters, lr=0.02)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)

num_epochs = 500
L = []
for ite in range(num_epochs):
    # model.train()  
    optimizer.zero_grad()

    # x_under  = F.interpolate(x,scale_factor=0.5)
    # ref_under = F.interpolate(ref,scale_factor=0.5)
    # x_loss, output, att, attAll  = model(x_under, ref_under)
    # loss = criterion(ref_under[0,:], x_loss)
    # output = F.interpolate(output.view(1,1,output.size(1),output.size(2)),scale_factor=2)

    # output = output[:,0,:,:]

    x_loss, output, att, attAll  = model(x, ref)
    loss = criterion(ref[0,:], x_loss)

    # loss = loss1 + loss2

    # loss = criterion(ref[0,:], output)
    loss.backward()
    optimizer.step()

    L.append(loss.detach().cpu().numpy())

    # plt.show()

    # x = output.view(1,1,output.shape[1],output.shape[2])
    # x = x_loss

    # plt.figure(1)
    # plt.imshow((output[0,:].cpu().detach().numpy()))

    # plt.figure()
    # plt.imshow(montage(att[0,:].cpu().detach().numpy()))

    # plt.figure()
    # plt.imshow(montage(feat[0,:].cpu().detach().numpy()))
    # # plt.draw()

    #     plt.show(block=False)

    # if ite%num_epochs==0 or ite==0:

    #     plt.figure(2)
    #     plt.imshow(montage(att1[0,:].cpu().detach().numpy()))

    #     plt.figure(6)
    #     plt.imshow((output[0,:].cpu().detach().numpy()))

    #     plt.figure(7)
    #     plt.imshow((x_loss[0,0,:].cpu().detach().numpy()))

    #     nr, nc = output.size(1),output.size(2)
    #     im = np.zeros((nr, nc, 3))
    #     im[..., 0] = ref[0,0,:].cpu().detach().numpy()
    #     im[..., 1] = output[0,:].cpu().detach().numpy()
    #     im[..., 2] = output[0,:].cpu().detach().numpy()
    #     plt.figure(9)
    #     plt.imshow(im)

    #     # plt.figure(8)
    #     # plt.imshow((x[0,0,:].cpu().detach().numpy()))

    #     plt.show(block=False)

plt.figure()
plt.imshow(montage(attAll[0,:].cpu().detach().numpy()))

nr, nc = output.size(1),output.size(2)
im = np.zeros((nr, nc, 3))
im[..., 0] = ref[0,0,:].cpu().detach().numpy()
im[..., 1] = x[0,:].cpu().detach().numpy()
im[..., 2] = x[0,:].cpu().detach().numpy()
plt.figure()
plt.imshow(im)

nr, nc = output.size(1),output.size(2)
im = np.zeros((nr, nc, 3))
im[..., 0] = ref[0,0,:].cpu().detach().numpy()
im[..., 1] = output[0,:].cpu().detach().numpy()
im[..., 2] = output[0,:].cpu().detach().numpy()
plt.figure()
plt.imshow(im)

plt.figure()
plt.imshow(ref[0,0,:].cpu().detach().numpy())

plt.figure()
plt.imshow(output[0,:].cpu().detach().numpy())

plt.figure()
plt.plot(L)
# plt.draw()

plt.show()

# output_Att_1 = F.softmax(output_Att_1, dim=1)

# plt.figure()
# plt.imshow(montage(output_Att_1[0,:].detach().numpy()))
# # plt.show()

# output_Att_1 = torch.argmax(output_Att_1, dim=1)

# plt.figure()
# plt.imshow(output_Att_1[0,:].detach().numpy())
# # plt.show()

# output_Att_1 = F.one_hot(output_Att_1, num_classes=nb_kernels)
# output_Att_1 = output_Att_1.permute(0,3,1,2)
# plt.figure()
# plt.imshow(montage(output_Att_1[0,:].detach().numpy()))


# plt.show()

# output_1.mean().backward()
# print(conv.weight)
# print(x)
# print(output_1[:,1,::])

# plt.figure()
# plt.imshow(output_Reg_1[0,2,::].detach().numpy())
# plt.show()



# plt.figure()
# plt.imshow(montage(output_Att_1[0,:].detach().numpy()))
# plt.show()

