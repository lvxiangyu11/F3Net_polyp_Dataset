import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from torch.autograd import Function

from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/')
parser.add_argument('--Anti_Alias_Downsample_use', type=bool, default=False, help='使用AntiAliasDownsampleLayer')

class BinarizedF(Function):
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        output = torch.where(input >= 1, a, input)
        return output

    def backward(self, output_grad):
        input, = self.saved_tensors
        input_abs = torch.abs(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs <= 1, ones, zeros)
        return input_grad

import imageio

def calDice(pred, mask, threshold=0.917647058823529):
    TP = (pred * mask).sum(dim=(2, 3))
    FN_und_FP_und_TP = (BinarizedF().forward(pred + mask)).sum(dim=(2, 3))
    dice = (2 * TP) / (TP + FN_und_FP_und_TP)
    return dice



def count(model, dataSetName):
    _data_name = dataSetName
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/RF3Net/{}/'.format(_data_name)

    with torch.no_grad():
        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        dice = 0.0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            out1u, out2u, out2r, out3r, out4r, out5r = model(image)
            res = out2u
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

            def countdic(pred, gt):
                pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                dice = calDice(pred=pred, mask=torch.from_numpy(gt).cuda())
                return dice

            out2u_ = F.upsample(torch.sigmoid(out2u), size=gt.shape, mode='bilinear', align_corners=False)

            res = out2u_
            dice += calDice(pred=res, mask=torch.from_numpy(gt).cuda())
            res = res.data.cpu().numpy().squeeze()
            imageio.imsave(save_path + name, res)

        return float(dice / test_loader.size)


opt = parser.parse_args()

# model = PraNet()

from lib.F3Net import F3Net
model = F3Net(opt)

from collections import OrderedDict
modelName = 'F3Net'
csvFileName = './results/F3Net/{}'.format(time.strftime("%Y.%m.%d_%X", time.localtime()))+'.csv'
csv = 'epoch,CVC-300,CVC-ClinicDB,Kvasir,CVC-ColonDB,ETIS-LaribPolypDB'
for i in range(69, 69+1, 1):
    print(i, end=' ')
    csv = csv+'\n'+str(i)

    state_dict = torch.load(opt.pth_path+modelName+'-'+str(i)+'.pth')
    # state_dict = torch.load(opt.pth_path+'PraNet'+'-'+str(i)+'.pth')
    new_state_dict = OrderedDict()
    # 处理一些，因为GPU集群运算导致，模型在保存时名称错误。
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        # name = "module."+k # add  `module.`
        # name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        csv = csv+','+str(count(model, _data_name))[0:6]

with open(csvFileName, 'w') as f:
    f.write(csv)