import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt

from .config import config

class SiameseAlexNet(nn.Module):
    def __init__(self, gpu_id, train=True):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
        self.corr_bias = nn.Parameter(torch.zeros(1))
        self.exemplar = None
        self.gpu_id = gpu_id
        self.pars_train =train

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar_imgs, exemplar_bboxes, instance_imgs, instance_bboxes = x
        if self.pars_train:
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz), instance_bboxes)
            with torch.cuda.device(self.gpu_id):
                self.train_gt = torch.from_numpy(gt).cuda()
                self.train_weight = torch.from_numpy(weight).cuda()
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz), instance_bboxes)
            with torch.cuda.device(self.gpu_id):
                self.valid_gt = torch.from_numpy(gt).cuda()
                self.valid_weight = torch.from_numpy(weight).cuda()
        if exemplar_imgs is not None and instance_imgs is not None:
            batch_size = exemplar_imgs.shape[0]
            exemplar_imgs = self.features(exemplar_imgs)
            instance_imgs = self.features(instance_imgs)
            score_map = []
            N, C, H, W = instance_imgs.shape
            instance_imgs = instance_imgs.view(1, -1, H, W)
            score = F.conv2d(instance_imgs, exemplar_imgs, groups=N) * config.response_scale \
                    + self.corr_bias
            return score.transpose(0, 1)
        elif exemplar_imgs is not None and instance_imgs is None:
            # inference used
            self.exemplar_imgs = self.features(exemplar_imgs)
            self.exemplar_imgs = torch.cat([self.exemplar_imgs for _ in range(3)], dim=0)
        else:
            # inference used we don't need to scale the reponse or add bias
            instance_imgs = self.features(instance_imgs)
            N, _, H, W = instance_imgs.shape
            instance_imgs = instance_imgs.view(1, -1, H, W)
            score = F.conv2d(instance_imgs, self.exemplar_imgs, groups=N)
            return score.transpose(0, 1)

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                    self.train_weight, reduction='sum') / config.train_batch_size # normalize the batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                    self.valid_weight, reduction='sum') / config.train_batch_size # normalize the batch_size

    def _create_gt_mask(self, shape, instance_bboxes):
        # same for all pairs
        h, w = shape
        instance_bboxes = instance_bboxes.cpu().data.numpy()
        sz_h = instance_bboxes[:,3]-instance_bboxes[:,1]
        sz_w = instance_bboxes[:,2]-instance_bboxes[:,0]
        scale =np.tanh((sz_h - sz_w) / sz_w)*0.5
        masks = np.zeros((config.train_batch_size,1,h,w))
        weights = np.zeros((config.train_batch_size,1,h,w))
        for idx in range(config.train_batch_size):
            y = np.arange(h, dtype=np.float32) - (h-1) / 2.
            x = np.arange(w, dtype=np.float32) - (w-1) / 2.
            y, x = np.meshgrid(y, x)
            dist = np.sqrt(x**2 + (y*(1+scale[idx]))**2)
            mask = np.zeros((h, w))
            mask[dist <= config.radius / config.total_stride] = 1
            mask = mask[np.newaxis, :, :]
            weight = np.ones_like(mask)
            weight[mask == 1] = 0.5 / (np.sum(mask == 1)+1)
            weight[mask == 0] = 0.5 / (np.sum(mask == 0)+1)
            masks[idx,:,:,:] = mask
            weights[idx,:,:,:] = weight
        return masks.astype(np.float32), weights.astype(np.float32)

def create_gt_mask_2(shape):
    # same for all pairs
    h, w = shape
    y = np.arange(h, dtype=np.float32) - (h-1) / 2.
    x = np.arange(w, dtype=np.float32) - (w-1) / 2.
    y, x = np.meshgrid(y, x)
    dist = np.sqrt(x**2 + y**2)
    mask = np.zeros((h, w))
    mask[dist <= 50] = 1
    mask = mask[np.newaxis, :, :]
    weights = np.ones_like(mask)
    weights[mask == 1] = 0.5 / np.sum(mask == 1)
    weights[mask == 0] = 0.5 / np.sum(mask == 0)
    mask = np.repeat(mask, 2, axis=0)[:, np.newaxis, :, :]
    return mask.astype(np.float32), weights.astype(np.float32)

def create_gt_mask_1(shape, instance_bboxes):
        # same for all pairs
        h, w = shape
        # instance_bboxes = instance_bboxes.cpu().data.numpy()
        sz_h = instance_bboxes[:,3]-instance_bboxes[:,1]
        sz_w = instance_bboxes[:,2]-instance_bboxes[:,0]
        scale =np.tanh((sz_h - sz_w) / sz_w)*0.5
        masks = np.zeros((4,1,h,w))
        weights = np.zeros((4,1,h,w))
        for idx in range(4):
            y = np.arange(h, dtype=np.float32) - (h-1) / 2.
            x = np.arange(w, dtype=np.float32) - (w-1) / 2.
            y, x = np.meshgrid(y, x)
            dist = np.sqrt((y*(1+scale[idx]))**2 + x**2)
            mask = np.zeros((h, w))
            mask[dist <= 2] = 1
            mask = mask[np.newaxis, :, :]
            weight = np.ones_like(mask)
            weight[mask == 1] = 0.5 / (np.sum(mask == 1)+1)
            weight[mask == 0] = 0.5 / (np.sum(mask == 0)+1)
            masks[idx,:,:,:] = mask
            weights[idx,:,:,:] = weight
        return masks.astype(np.float32), weights.astype(np.float32)


if __name__=='__main__':
    shape = (15,15)
    sz = np.array([[586,376,702,693],[840,167,1006,243],[638,145,885,460],[10,10,250,250]])
    a,b = create_gt_mask_1(shape, sz)
    temp =a[1][0]
    x = np.arange(0, 15, 1)  # len = 11
    y = np.arange(0, 15, 1)  # len = 7
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.pcolormesh(x, y, temp)
    #path ='F:\\SiamFC-PyTorch\\img\\'+str(0)+'.jpg'
    #plt.savefig(path)
    plt.show()
