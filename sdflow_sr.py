import os, argparse
import numpy as np
import torch
import modules as m
from PIL import Image
from glob import glob
from models import HRFlow


parser = argparse.ArgumentParser()
parser.add_argument('--lr_imgs_path', type=str, help='path to the LR images folder', required=True)
parser.add_argument('--output_path', type=str, help='output folder for the SR images', required=True)
parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained models folder', required=True)
parser.add_argument('--tau', type=float, help='sampling temperature', default=0.8)

args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    lr_imgs_path = args.hr_imgs_path
    output_path = args.output_path
    checkpoint_path = args.pretrained_model_path
    TAU = args.tau

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    LREncoder = m.LRImageEncoderFM(3, 3, is_downscale=False, with_uncertainty=False).cuda()
    HRNet = HRFlow().cuda()

    LREncoder.load_state_dict(torch.load(f'./{checkpoint_path}/lrencoder.pth', map_location='cpu'))
    HRNet.load_state_dict(torch.load(f'./{checkpoint_path}/hrnet.pth', map_location='cpu'))

    for lr_file in glob(os.path.join(lr_imgs_path, '*.png')):
        print(lr_file)
        name = os.path.basename(lr_file)
        lr_img = Image.open(lr_file).convert('RGB')
        lr_img = np.array(lr_img) / 255.
        lr_img = torch.from_numpy(lr_img.transpose(2, 0, 1)).float().unsqueeze(0)
        lr_img = lr_img.cuda(non_blocking=True)

        lr_img.mul_(2).sub_(1)
        z_lr = LREncoder(lr_img)
        lr2hr, _ = HRNet(z_lr, None, tau=TAU, reverse=True)

        lr2hr = torch.where(torch.isnan(lr2hr), torch.rand_like(lr2hr), lr2hr)
        lr2hr.add_(1).mul_(0.5).clamp_(0, 1)
    
        lr2hr_np = lr2hr.data.cpu().numpy().squeeze().transpose(1, 2, 0)
        img_sr = Image.fromarray(np.uint8(lr2hr_np * 255))
        img_sr.save(os.path.join(output_path, f'{name}'))