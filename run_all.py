"""
SeIF - Visu attēlu rekonstrukcija
"""

import subprocess
import os
import sys
import time

# VISI attēlu numuri
all_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Mape
project_dir = r"D:\Download\Desktop\3D_Sejas"

def create_temp_script(number):
    """Izveido pagaidu Python skriptu konkrētam attēlam"""
    
    temp_script = f'''import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from lib.options import BaseOptions
from lib.options2 import BaseOptions as BaseOptions2
from lib.model import *
from lib.train_util import *
from Constants import consts
from lib.model2 import HGPIFuNet as SemanticNet
import torchvision.transforms as transforms

opt = BaseOptions().parse()
opt2 = BaseOptions2().parse()

def rotateY_by_view(view_id):
    angle = np.radians(-view_id)
    ry = np.array([[np.cos(angle), 0., np.sin(angle)],
                   [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    ry = np.transpose(ry)
    return ry

to_tensor = transforms.Compose([
    transforms.Resize(opt.loadSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    device = torch.device('cpu')
    projection_mode = 'orthogonal'
    
    number = {number}
    print(f"Processing image: {{number}}")
    
    # File paths
    image_path = f'demo/{{number}}.png'
    image_mask_path = f'demo/{{number}}_mask.png'
    normal_path = f'demo/{{number}}_normal.png'
    save_file = f'demo/{{number}}.obj'
    
    if not os.path.exists(image_path):
        print(f"ERROR: {{image_path}} not found!")
        return False
    if not os.path.exists(image_mask_path):
        print(f"ERROR: {{image_mask_path}} not found!")
        return False
    if not os.path.exists(normal_path):
        print(f"ERROR: {{normal_path}} not found!")
        return False
    
    # Load models
    netG_SeIF = HGPIFuNet(opt, projection_mode)
    netG_SeIF.to(device)
    netG_Semantic = SemanticNet(opt2, projection_mode)
    netG_Semantic.to(device)
    
    model_semantic_path = r'checkpoints/model_Semantic/netG_epoch_6_293299'
    netG_Semantic.load_state_dict(torch.load(model_semantic_path, map_location=device))
    netG_Semantic.eval()
    
    model_SeIF_path = 'checkpoints/model_SeIF/netG_epoch_3_293566'
    netG_SeIF.load_state_dict(torch.load(model_SeIF_path, map_location=device))
    netG_SeIF.eval()
    
    try:
        render_list = []
        normal_list = []
        calib_list = []
        extrinsic_list = []
        mask_list = []

        # load mask
        mask_data = np.round((cv2.imread(image_mask_path)[:, :, 0]).astype(np.float32) / 255.)
        mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32)
        mask_data_padded[:,
        mask_data_padded.shape[0] // 2 - min(mask_data.shape) // 2:mask_data_padded.shape[0] // 2 + min(
            mask_data.shape) // 2] = mask_data

        mask_data_padded = cv2.resize(mask_data_padded, (opt.loadSize, opt.loadSize),
                                      interpolation=cv2.INTER_NEAREST)
        mask_data_padded = Image.fromarray(mask_data_padded)

        # load image
        image = cv2.imread(image_path)[:, :, ::-1]
        normal_image = cv2.imread(normal_path)[:, :, ::-1]

        image_padded = np.zeros((max(image.shape), max(image.shape), 3), np.uint8)
        normal_image_padded = np.zeros((max(normal_image.shape), max(normal_image.shape), 3), np.uint8)
        image_padded[:,
        image_padded.shape[0] // 2 - min(image.shape[:2]) // 2:image_padded.shape[0] // 2 + min(
            image.shape[:2]) // 2,
        :] = image
        normal_image_padded[:,
        normal_image_padded.shape[0] // 2 - min(normal_image.shape[:2]) // 2:normal_image_padded.shape[
                                                                                 0] // 2 + min(
            normal_image.shape[:2]) // 2, :] = normal_image

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
        viewRot = rotateY_by_view(view_id=0)
        extrinsic[:3, :3] = viewRot.T

        mask_data_padded = transforms.ToTensor()(mask_data_padded).float()
        mask_list.append(mask_data_padded)

        image_padded = to_tensor(image_padded)
        normal_image_padded = to_tensor(normal_image_padded)
        image_padded = mask_data_padded.expand_as(image_padded) * image_padded
        normal_image_padded = mask_data_padded.expand_as(normal_image_padded) * normal_image_padded
        render_list.append(image_padded)
        normal_list.append(normal_image_padded)

        intrinsic = np.matmul(trans_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

        extrinsic = torch.Tensor(extrinsic).float()
        calib_list.append(calib)
        extrinsic_list.append(extrinsic)

        data = {{'img': torch.stack(render_list, dim=0),
                'normal': torch.stack(normal_list, dim=0),
                'calib': torch.stack(calib_list, dim=0),
                'extrinsic': torch.stack(extrinsic_list, dim=0),
                'mask': torch.stack(mask_list, dim=0)
                }}
        
        gen_mesh(opt, netG_SeIF.module if len(opt.gpu_ids) > 1 else netG_SeIF, netG_Semantic, device, data, save_file)
        
        print(f"SUCCESS: {{save_file}} saved!")
        return True
        
    except Exception as e:
        print(f"ERROR: {{e}}")
        return False

if __name__ == '__main__':
    result = main()
    exit(0 if result else 1)
'''
    
    script_path = os.path.join(project_dir, f'temp_{number}.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(temp_script)
    return script_path

def main():
    print("=" * 60)
    print("SeIF - All 16 Images Reconstruction")
    print("=" * 60)
    print(f"Images to process: {all_numbers}")
    print("-" * 60)
    
    os.chdir(project_dir)
    
    success_list = []
    failed_list = []
    
    for i, number in enumerate(all_numbers, 1):
        print(f"\n[{i}/16] Processing {number}.png...")
        
        temp_script = create_temp_script(number)
        
        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                timeout=600,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  OK - {number}.obj created!")
                success_list.append(number)
            else:
                print(f"  FAILED - {number}.obj")
                if result.stderr:
                    print(f"  Error: {result.stderr[:300]}")
                failed_list.append(number)
                
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT - {number}.obj (10 min limit)")
            failed_list.append(number)
        except Exception as e:
            print(f"  ERROR - {number}.obj: {e}")
            failed_list.append(number)
        
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        time.sleep(3)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Successful: {len(success_list)}/16 - {success_list}")
    if failed_list:
        print(f"Failed: {len(failed_list)}/16 - {failed_list}")
    print(f"\nResults in: {project_dir}\\demo\\")
    print("=" * 60)

if __name__ == '__main__':
    main()