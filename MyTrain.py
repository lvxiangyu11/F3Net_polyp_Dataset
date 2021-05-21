import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F


def structure_loss(pred, mask):
    mask = F.interpolate(mask, size=pred.size()[2:], mode='bilinear')
    edge = torch.abs(F.max_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    weit = 1 * edge + 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25, 1.5]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            out1u, out2u, out2r, out3r, out4r, out5r = model(images)
            #
            loss1u = structure_loss(out1u, gts)
            loss2u = structure_loss(out2u, gts)
            #
            loss2 = structure_loss(out2r, gts)
            loss3 = structure_loss(out3r, gts)
            loss4 = structure_loss(out4r, gts)
            loss5 = structure_loss(out5r, gts)
            # loss = loss1u + loss2u * 3 + loss2 + loss3 + loss4 + loss5 * 2
            loss = (loss1u + loss2u) / 2 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        # ---- train visualization ----
        if i % 200 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    #if (epoch+1) % 10 == 0:
    if True:
        torch.save(model.state_dict(), save_path + 'RF3Net-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'RF3Net-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=3e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset/AuthorData', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='PraNet_Res2Net')
    parser.add_argument('--Anti_Alias_Downsample_use', type=bool, default=False, help='使用AntiAliasDownsampleLayer')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = PraNet().cuda()
    #
    from lib.F3Net import F3Net
    model = F3Net(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)

