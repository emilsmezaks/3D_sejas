
import cv2
import numpy as np
import os

def prepare_using_mask(demo_dir='./demo', filename='6', target_size=512):
    
    img_path = os.path.join(demo_dir, f'{filename}.png')
    mask_path = os.path.join(demo_dir, f'{filename}_mask.png')
    normal_path = os.path.join(demo_dir, f'{filename}_normal.png')
    
    print(f"SAGATAVO ATTĒLU: {filename}")
 
    if not os.path.exists(mask_path):
        print(f"Maska nav atrasta{mask_path}")
        return False
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Nevar ielādēt masku{mask_path}")
        return False
    
    mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    face_region = mask_resized > 128
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
        
            img_resized = cv2.resize(img, (target_size, target_size))
            
            img_result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
     
            img_result[face_region] = img_resized[face_region]
            
            cv2.imwrite(img_path, img_result)
            print(f"{filename}.png: seja saglabāta")
        else:
            print(f"Nevar{img_path}")
    else:
        print(f"Nav atrasts{img_path}")
    

    if os.path.exists(normal_path):
        normal = cv2.imread(normal_path)
        if normal is not None:
        
            normal_resized = cv2.resize(normal, (target_size, target_size))
            
            
            normal_result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            normal_result[:, :] = [128, 128, 128] 
        
            normal_result[face_region] = normal_resized[face_region]
            
            cv2.imwrite(normal_path, normal_result)
            print(f"{filename}_normal.png")
            
            normal_float = normal_result.astype(np.float32) / 255.0
            normal_float = normal_float * 2.0 - 1.0  # [0,255] -> [-1,1]
            npy_path = normal_path.replace('.png', '.npy')
            np.save(npy_path, normal_float)
            print(f"{filename}_normal.npy saglabāts")
        else:
            print(f"Nevar{normal_path}")
    else:
        print(f"Nav atrasts{normal_path}")
    
    return True


if __name__ == "__main__":
    prepare_using_mask(demo_dir='./demo', filename='6', target_size=512)