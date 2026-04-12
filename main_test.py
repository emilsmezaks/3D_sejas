import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from lib.options import BaseOptions
from lib.options2 import BaseOptions as BaseOptions2
from lib.model import *
from lib.train_util import *
from Constants import consts
from lib.model2 import HGPIFuNet as SemanticNet
import torchvision.transforms as transforms
import cv2
from PIL import Image

# 🔥 FORCE CPU
torch.cuda.is_available = lambda: False

opt = BaseOptions().parse()
opt2 = BaseOptions2().parse()

def rotateY_by_view(view_id):
    angle = np.radians(-view_id)
    ry = np.array([[np.cos(angle), 0., np.sin(angle)],
                   [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.transpose(ry)

to_tensor = transforms.Compose([
    transforms.Resize(opt.loadSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main(opt):
    image_path = 'demo/1.png'
    image_mask_path = 'demo/1_mask.png'
    normal_path = 'demo/1_normal.png'
    save_file = 'demo/1.obj'

    # ✅ CPU DEVICE
    device = torch.device('cpu')
    projection_mode = 'orthogonal'

    with torch.no_grad():
        netG_SeIF = HGPIFuNet(opt, projection_mode).to(device)
        netG_Semantic = SemanticNet(opt2, projection_mode).to(device)

        model_semantic_path = r'checkpoints/model_Semantic/netG_epoch_6_293299'
        netG_Semantic.load_state_dict(torch.load(model_semantic_path, map_location=device))
        netG_Semantic.eval()

        model_SeIF_path = 'checkpoints/model_SeIF/netG_epoch_3_293566'
        netG_SeIF.load_state_dict(torch.load(model_SeIF_path, map_location=device))
        netG_SeIF.eval()

        if os.path.exists(image_path):
            if not os.path.exists(save_file):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)

            render_list = []
            normal_list = []
            calib_list = []
            extrinsic_list = []
            mask_list = []

            # mask
            mask_data = np.round((cv2.imread(image_mask_path)[:, :, 0]).astype(np.float32) / 255.)
            mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32)
            mask_data_padded[:, mask_data_padded.shape[0] // 2 - min(mask_data.shape) // 2:
                                mask_data_padded.shape[0] // 2 + min(mask_data.shape) // 2] = mask_data

            mask_data_padded = cv2.resize(mask_data_padded, (opt.loadSize, opt.loadSize),
                                          interpolation=cv2.INTER_NEAREST)
            mask_data_padded = Image.fromarray(mask_data_padded)

            # images
            image = cv2.imread(image_path)[:, :, ::-1]
            normal_image = cv2.imread(normal_path)[:, :, ::-1]

            image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)
            normal_image_padded = np.zeros((max(normal_image.shape), max(normal_image.shape), 3), np.uint8)

            image_padded[:, image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:
                            image_padded.shape[0] // 2 + min(image.shape[:2]) // 2, :] = image

            normal_image_padded[:, normal_image_padded.shape[0] // 2 - min(normal_image.shape[:2]) // 2:
                                   normal_image_padded.shape[0] // 2 + min(normal_image.shape[:2]) // 2, :] = normal_image

            image_padded = cv2.resize(image_padded, (opt.loadSize, opt.loadSize))
            normal_image_padded = cv2.resize(normal_image_padded, (opt.loadSize, opt.loadSize))

            image_padded = Image.fromarray(image_padded)
            normal_image_padded = Image.fromarray(normal_image_padded)

            trans_intrinsic = np.identity(4)
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = 1. / consts.h_normalize_half
            scale_intrinsic[1, 1] = -1. / consts.h_normalize_half
            scale_intrinsic[2, 2] = -1. / consts.h_normalize_half

            extrinsic = np.identity(4)
            viewRot = rotateY_by_view(0)
            extrinsic[:3, :3] = viewRot.T

            mask_tensor = transforms.ToTensor()(mask_data_padded).float()
            mask_list.append(mask_tensor)

            image_tensor = to_tensor(image_padded)
            normal_tensor = to_tensor(normal_image_padded)

            image_tensor = mask_tensor.expand_as(image_tensor) * image_tensor
            normal_tensor = mask_tensor.expand_as(normal_tensor) * normal_tensor

            render_list.append(image_tensor)
            normal_list.append(normal_tensor)

            intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

            extrinsic = torch.Tensor(extrinsic).float()

            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

            data = {
                'img': torch.stack(render_list, dim=0),
                'normal': torch.stack(normal_list, dim=0),
                'calib': torch.stack(calib_list, dim=0),
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask': torch.stack(mask_list, dim=0)
            }

            gen_mesh(
                opt,
                netG_SeIF,
                netG_Semantic,
                device,
                data,
                save_file
            )

if __name__ == '__main__':
    main(opt)