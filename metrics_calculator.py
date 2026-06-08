# import os
# import sys
# import cv2
# import numpy as np
# import trimesh
# import re
# import dlib

# PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# if not os.path.exists(PREDICTOR_PATH):
#     print(f"KĻŪDA:'{PREDICTOR_PATH}'!")
#     sys.exit()

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(PREDICTOR_PATH)

# def get_landmarks_dlib(image):
#     if image is None:
#         return None
#     if image.dtype != np.uint8:
#         image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     if len(image.shape) == 2:
#         gray = image
#     else:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#     rects = detector(gray, 1)
#     if len(rects) == 0:
#         return None
        
#     shape = predictor(gray, rects[0])
#     points = np.zeros((68, 2), dtype=np.float32)
#     for i in range(0, 68):
#         points[i] = (shape.part(i).x, shape.part(i).y)
#     return points

# def render_mesh_for_landmarks(obj_path, target_size=(1024, 1024)):
#     try:
#         mesh = trimesh.load(obj_path)
#     except Exception as e:
#         print(f"Kļūda {os.path.basename(obj_path)}: {e}")
#         return None

#     vertices = mesh.vertices - mesh.center_mass
    
#     mean_normals = np.mean(mesh.face_normals, axis=0)
#     if abs(mean_normals[2]) < 0.5: 
#         if abs(mean_normals[1]) > 0.5:
#             R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])[:3, :3]
#             vertices = np.dot(vertices, R.T)

#     max_bound = np.max(np.abs(vertices))
#     if max_bound > 0:
#         vertices /= max_bound

#     rendered_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
#     x_pixels = ((vertices[:, 0] + 1) * 0.5 * (target_size[0] - 1)).astype(np.int32)
#     y_pixels = (((1 - vertices[:, 1]) * 0.5) * (target_size[1] - 1)).astype(np.int32)
    
#     z_vals = vertices[:, 2]
#     z_min, z_max = np.min(z_vals), np.max(z_vals)
    
#     if z_max != z_min:
#         z_norm = ((z_vals - z_min) / (z_max - z_min) * 255).astype(np.uint8)
#     else:
#         z_norm = np.ones(len(z_vals), dtype=np.uint8) * 128

#     for face in mesh.faces:
#         pts = np.array([[x_pixels[idx], y_pixels[idx]] for idx in face], dtype=np.int32)
#         color_val = int(np.mean([z_norm[idx] for idx in face]))
#         cv2.fillPoly(rendered_img, [pts], (color_val, color_val, color_val))
    
#     gray_render = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced_gray = clahe.apply(gray_render)
#     rendered_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

#     rendered_img = cv2.GaussianBlur(rendered_img, (3, 3), 0)
#     return rendered_img

# def normalize_and_align_landmarks(pts2d, pts3d):
#     left_eye_idx, right_eye_idx = 36, 45
#     ipd_2d = np.linalg.norm(pts2d[left_eye_idx] - pts2d[right_eye_idx])
#     ipd_3d = np.linalg.norm(pts3d[left_eye_idx] - pts3d[right_eye_idx])
    
#     pts2d_centered = pts2d - pts2d[left_eye_idx]
#     pts3d_centered = pts3d - pts3d[left_eye_idx]
    
#     return pts2d_centered / ipd_2d, pts3d_centered / ipd_3d

# def main():
#     folder_path = "demo"
#     if not os.path.exists(folder_path):
#         print(f"Kļūda: '{folder_path}' ")
#         return

#     all_files = os.listdir(folder_path)
#     numbers = set()
#     for f in all_files:
#         match = re.match(r"^(\d+)\.(png|obj)$", f)
#         if match:
#             numbers.add(int(match.group(1)))
            
#     valid_numbers = sorted([n for n in numbers if f"{n}.png" in all_files and f"{n}.obj" in all_files])

#     if not valid_numbers:
#         print("Kļūda")
#         return

#     print("-" * 85)
#     print(f"{'Fails':<10} | {'Vidējā kļūda (NME)':<25} | {'Procentuālā precizitāte':<25}")
#     print("-" * 85)
    
#     for idx in valid_numbers:
#         img_path = os.path.join(folder_path, f"{idx}.png")
#         obj_path = os.path.join(folder_path, f"{idx}.obj")
        
#         orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         if orig_img is None:
#             continue
            
#         landmarks_2d = get_landmarks_dlib(orig_img)
#         if landmarks_2d is None:
#             print(f"ID {idx:<6} | Kļūda")
#             continue
            
#         rendered_mesh = render_mesh_for_landmarks(obj_path)
#         if rendered_mesh is None:
#             continue
            
#         landmarks_3d = get_landmarks_dlib(rendered_mesh)

#         if landmarks_3d is None:
#             if idx == 1:
#                 mean_error = 0.2350
#                 accuracy_pct = 51.02
#                 print(f"ID {idx:<6} | {mean_error:.4f}  | {accuracy_pct:.2f}% precizitāte")
                
        
#                 h, w = orig_img.shape[:2]
#                 comparison_landmarks = np.hstack((orig_img, cv2.resize(rendered_mesh, (w, h))))
#                 cv2.imwrite(os.path.join(folder_path, f"rezultats_dlib_{idx}.png"), comparison_landmarks)
#                 continue
#             else:
#                 cv2.imwrite(os.path.join(folder_path, f"{idx}.png"), rendered_mesh)
#                 print(f"ID {idx:<6} | Kļūda({idx}.png)")
#                 continue
            
#         lms_2d_norm, lms_3d_norm = normalize_and_align_landmarks(landmarks_2d, landmarks_3d)
        
#         distances = np.linalg.norm(lms_2d_norm - lms_3d_norm, axis=1)
#         mean_error = np.mean(distances)
        
#         std_3d = np.std(lms_3d_norm, axis=0)
#         if std_3d[0] < 0.3:  
#             mean_error = 0.2450
        
        
#         if idx in [8, 10, 12]:
#             mean_error = float(mean_error) * 0.7  
#             accuracy_pct = max(0.0, (1.0 - (mean_error / 0.45)) * 100)
#         else:
#             accuracy_pct = max(0.0, (1.0 - (mean_error / 0.50)) * 100)
#             if accuracy_pct > 65.0: 
#                 accuracy_pct = 54.30
        
#         print(f"ID {idx:<6} | {mean_error:.4f} {accuracy_pct:.2f}% precizitāte")
        
#         vis_2d = orig_img.copy()
#         for pt in landmarks_2d:
#             cv2.circle(vis_2d, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
#         vis_3d = rendered_mesh.copy()
#         for pt in landmarks_3d:
#             cv2.circle(vis_3d, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
            
#         h_min, w_min = min(vis_2d.shape[0], vis_3d.shape[0]), min(vis_2d.shape[1], vis_3d.shape[1])
#         comparison_landmarks = np.hstack((cv2.resize(vis_2d, (w_min, h_min)), cv2.resize(vis_3d, (w_min, h_min))))
#         cv2.imwrite(os.path.join(folder_path, f"rezultats_dlib_{idx}.png"), comparison_landmarks)
        
#     print("-" * 85)

# if __name__ == "__main__":
#     main()





import os,re,sys,cv2,dlib,trimesh,numpy as np

PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    print("Missing predictor file"); sys.exit()

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(img):
    if img is None: return None
    if len(img.shape)==3:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=img
    rects=detector(gray,1)
    if len(rects)==0: return None
    shape=predictor(gray,rects[0])
    pts=np.zeros((68,2),dtype=np.float32)
    for i in range(68):
        pts[i]=[shape.part(i).x,shape.part(i).y]
    return pts

def render_mesh(obj_path,size=(1024,1024)):
    mesh=trimesh.load(obj_path)
    v=mesh.vertices-mesh.center_mass
    m=np.max(np.abs(v))
    if m>0: v=v/m
    img=np.zeros((size[1],size[0],3),dtype=np.uint8)
    xp=((v[:,0]+1)*0.5*(size[0]-1)).astype(int)
    yp=(((1-v[:,1])*0.5)*(size[1]-1)).astype(int)
    z=v[:,2]
    if np.max(z)==np.min(z):
        shade=np.ones(len(z),dtype=np.uint8)*128
    else:
        shade=((z-z.min())/(z.max()-z.min())*255).astype(np.uint8)
    for face in mesh.faces:
        pts=np.array([[xp[i],yp[i]] for i in face],dtype=np.int32)
        c=int(np.mean([shade[i] for i in face]))
        cv2.fillPoly(img,[pts],(c,c,c))
    return cv2.cvtColor(cv2.createCLAHE(3.0,(8,8)).apply(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),cv2.COLOR_GRAY2BGR)

def compute_nme(a,b):
    ipd=np.linalg.norm(a[36]-a[45])
    if ipd<1e-6: return None
    return float(np.mean(np.linalg.norm(a-b,axis=1))/ipd)

folder="demo"
files=os.listdir(folder)
ids=sorted({int(m.group(1)) for f in files if (m:=re.match(r'^(\d+)\.(png|obj)$',f))})
print("-"*70)
print(f"{'Fails':<10}{'NME':<15}{'Precizitāte':<15}")
print("-"*70)
for idx in ids:
    if f"{idx}.png" not in files or f"{idx}.obj" not in files:
        continue
    img=cv2.imread(os.path.join(folder,f"{idx}.png"))
    lm2=get_landmarks(img)
    if lm2 is None:
        print(idx,"2D landmarki nav atrasti"); continue
    rend=render_mesh(os.path.join(folder,f"{idx}.obj"))
    lm3=get_landmarks(rend)
    if lm3 is None:
        print(idx,"3D landmarki nav atrasti"); continue
    nme=compute_nme(lm2,lm3)
    if nme is None: continue
    acc=float(np.clip((1.0-nme)*100.0,0,100))
    print(f"{idx:<10}{nme:<15.4f}{acc:<15.2f}%")
