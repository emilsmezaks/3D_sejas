# import os
# import sys
# import cv2
# import numpy as np
# import trimesh
# import re
# import dlib

# PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
# if not os.path.exists(PREDICTOR_PATH):
#     print("Missing predictor file"); sys.exit()

# detector=dlib.get_frontal_face_detector()
# predictor=dlib.shape_predictor(PREDICTOR_PATH)

# def get_landmarks(img):
#     if img is None: return None
#     if len(img.shape)==3:
#         gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     else:
#         gray=img
#     rects=detector(gray,1)
#     if len(rects)==0: return None
#     shape=predictor(gray,rects[0])
#     pts=np.zeros((68,2),dtype=np.float32)
#     for i in range(68):
#         pts[i]=[shape.part(i).x,shape.part(i).y]
#     return pts

# def render_mesh(obj_path,size=(1024,1024)):
#     mesh=trimesh.load(obj_path)
#     v=mesh.vertices-mesh.center_mass
#     m=np.max(np.abs(v))
#     if m>0: v=v/m
#     img=np.zeros((size[1],size[0],3),dtype=np.uint8)
#     xp=((v[:,0]+1)*0.5*(size[0]-1)).astype(int)
#     yp=(((1-v[:,1])*0.5)*(size[1]-1)).astype(int)
#     z=v[:,2]
#     if np.max(z)==np.min(z):
#         shade=np.ones(len(z),dtype=np.uint8)*128
#     else:
#         shade=((z-z.min())/(z.max()-z.min())*255).astype(np.uint8)
#     for face in mesh.faces:
#         pts=np.array([[xp[i],yp[i]] for i in face],dtype=np.int32)
#         c=int(np.mean([shade[i] for i in face]))
#         cv2.fillPoly(img,[pts],(c,c,c))
#     return cv2.cvtColor(cv2.createCLAHE(3.0,(8,8)).apply(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),cv2.COLOR_GRAY2BGR)

# def compute_nme(a,b):
#     ipd=np.linalg.norm(a[36]-a[45])
#     if ipd<1e-6: return None
#     return float(np.mean(np.linalg.norm(a-b,axis=1))/ipd)

# folder="demo"
# files=os.listdir(folder)
# ids=sorted({int(m.group(1)) for f in files if (m:=re.match(r'^(\d+)\.(png|obj)$',f))})
# print("-"*70)
# print(f"{'Fails':<10}{'NME':<15}{'Precizitāte':<15}")
# print("-"*70)
# for idx in ids:
#     if f"{idx}.png" not in files or f"{idx}.obj" not in files:
#         continue
#     img=cv2.imread(os.path.join(folder,f"{idx}.png"))
#     lm2=get_landmarks(img)
#     if lm2 is None:
#         print(idx,"2D landmarki nav atrasti"); continue
#     rend=render_mesh(os.path.join(folder,f"{idx}.obj"))
#     lm3=get_landmarks(rend)
#     if lm3 is None:
#         print(idx,"3D landmarki nav atrasti"); continue
#     nme=compute_nme(lm2,lm3)
#     if nme is None: continue
#     acc=float(np.clip((1.0-nme)*100.0,0,100))
#     print(f"{idx:<10}{nme:<15.4f}{acc:<15.2f}%")




import os
import cv2
import dlib
import trimesh
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                   dtype=np.float32)
    return pts


def render_mesh(obj_path, size=(1024,1024)):
    mesh = trimesh.load(obj_path)
    v = mesh.vertices.copy()
    v -= v.mean(axis=0)

    m = np.abs(v).max()
    if m > 0:
        v /= m

    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    x = ((v[:,0]+1)/2*(size[0]-1)).astype(int)
    y = (((1-v[:,1])/2)*(size[1]-1)).astype(int)

    z = v[:,2]
    z = ((z-z.min())/(z.max()-z.min()+1e-8)*255).astype(np.uint8)

    for f in mesh.faces:
        pts = np.array([[x[i],y[i]] for i in f], np.int32)
        c = int(np.mean(z[f]))
        cv2.fillConvexPoly(img, pts, (c,c,c))

    img = cv2.GaussianBlur(img,(3,3),0)
    return img


def normalize_landmarks(pts):
    left = pts[36]
    right = pts[45]
    ipd = np.linalg.norm(left-right)
    pts = pts-left
    return pts/(ipd+1e-8)


def compute_nme(gt, pred):
    gt = normalize_landmarks(gt)
    pred = normalize_landmarks(pred)
    d = np.linalg.norm(gt-pred, axis=1)
    return float(np.mean(d))


def compute_accuracy(gt, pred, threshold=0.08):
    gt = normalize_landmarks(gt)
    pred = normalize_landmarks(pred)
    d = np.linalg.norm(gt-pred, axis=1)
    correct = np.sum(d < threshold)
    return 100.0 * correct / len(d)


def main(folder="demo"):
    print(f'{"Fails":<10} {"NME":<12} {"Precizitāte":<15}')
    print("-"*40)

    for f in sorted(os.listdir(folder)):
        if not f.endswith(".png"):
            continue
        stem = os.path.splitext(f)[0]
        obj = os.path.join(folder, stem + ".obj")
        if not os.path.exists(obj):
            continue

        img = cv2.imread(os.path.join(folder,f))
        lm2d = get_landmarks(img)
        if lm2d is None:
            continue

        rend = render_mesh(obj)
        lm3d = get_landmarks(rend)
        if lm3d is None:
            continue

        nme = compute_nme(lm2d, lm3d)
        acc = compute_accuracy(lm2d, lm3d)

        print(f"{stem:<10} {nme:<12.4f} {acc:>7.2f}%")

if __name__ == "__main__":
    main()
