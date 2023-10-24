import os, argparse
import numpy as np
import torch
from PIL import Image
from glob import glob
from models import HRFlow, ContentFlow, DegFlow

parser = argparse.ArgumentParser()
parser.add_argument('--hr_imgs_path', type=str, help='path to the HR images folder', required=True)
parser.add_argument('--output_path', type=str, help='output folder for the downscaled images', required=True)
parser.add_argument('--pretrained_model_path', type=str, help='path to the pretrained models folder', required=True)
parser.add_argument('--tau', type=float, help='sampling temperature', default=0.8)

args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    hr_imgs_path = args.hr_imgs_path
    output_path = args.output_path
    checkpoint_path = args.pretrained_model_path
    TAU = args.tau

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ContentNet = ContentFlow().cuda()
    HRNet = HRFlow().cuda()
    DegNet = DegFlow(3, 3, 16).cuda()

    ContentNet.load_state_dict(torch.load(f'./{checkpoint_path}/contentnet.pth', map_location='cpu'))
    HRNet.load_state_dict(torch.load(f'./{checkpoint_path}/hrnet.pth', map_location='cpu'))
    DegNet.load_state_dict(torch.load(f'./{checkpoint_path}/degnet.pth', map_location='cpu'))

    for hr_file in glob(os.path.join(hr_imgs_path, '*.png')):
        print(hr_file)
        name = os.path.basename(hr_file)
        hr_img = Image.open(hr_file).convert('RGB')
        w, h = hr_img.size
        assert w % 4 == 0
        assert h % 4 == 0
        new_w = (w + 8) // 8 * 8 if w % 8 != 0 else w
        new_h = (h + 8) // 8 * 8 if h % 8 != 0 else h
        new_hr_img = Image.new(hr_img.mode, (new_w, new_h), (124, 116, 104))
        new_hr_img.paste(hr_img, (0, 0))
        
        hr_img = np.array(new_hr_img) / 255.
        hr_img = torch.from_numpy(hr_img.transpose(2, 0, 1)).float().unsqueeze(0)
        hr_img = hr_img.cuda(non_blocking=True)

        hr_img.mul_(2).sub_(1)
        z_hr, _ = HRNet(hr_img, torch.zeros(1,).cuda())
        deg, _ = DegNet(None, None, z_hr, tau=TAU, reverse=True)
        hr2lr, _ = ContentNet(z_hr + deg, None, reverse=True)
        hr2lr = torch.where(torch.isnan(hr2lr), torch.rand_like(hr2lr), hr2lr)
        hr2lr.add_(1).mul_(0.5).clamp_(0, 1)

        hr2lr_np = hr2lr.data.cpu().numpy().squeeze().transpose(1, 2, 0)
        img_ds = Image.fromarray(np.uint8(hr2lr_np * 255))
        img_ds = img_ds.crop((0, 0, w // 4, h // 4))
        img_ds.save(os.path.join(output_path, f'{name}'))
