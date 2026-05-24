# import cv2
# import numpy as np
# import trimesh
# import os
# import glob
# from skimage.metrics import structural_similarity as ssim


# DEMO_DIR = "demo"
# OUTPUT_DIR = "evaluation_results"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# image_files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.png")))
# obj_files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.obj")))

# obj_map = {os.path.splitext(os.path.basename(f))[0]: f for f in obj_files}

# pairs = []
# for image_path in image_files:
#     base = os.path.splitext(os.path.basename(image_path))[0]
#     if base in obj_map:
#         pairs.append((image_path, obj_map[base]))

# if len(pairs) == 0:
#     raise RuntimeError("")

# print(f"Found {len(pairs)}")
# print("=" * 60)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# results_summary = []

# for image_path, obj_path in pairs:
#     print(f"\nProcessing: {os.path.basename(image_path)}")
#     print(f"Mesh: {os.path.basename(obj_path)}")
#     print("-" * 40)

#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Could not load image: {image_path}")
#         continue

#     height, width = image.shape[:2]
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         print(f"Face not detected")
#         continue

#     (x, y, w, h) = faces[0]
#     print(f"Face detected: x={x}, y={y}, w={w}, h={h}")

#     mask_original = np.zeros((height, width), dtype=np.uint8)
#     cv2.rectangle(mask_original, (x, y), (x + w, y + h), 255, -1)
#     area_original = mask_original.sum() / 255.0

#     try:
#         mesh = trimesh.load(obj_path)
#         if mesh.is_empty:
#             print(f"Mesh is empty")
#             continue
#         print(f"  Mesh vertices: {len(mesh.vertices):,}")
#         print(f"  Mesh faces: {len(mesh.faces):,}")
#     except Exception as e:
#         print(f"{e}")
#         continue


#     mesh_min = mesh.vertices.min(axis=0)
#     mesh_max = mesh.vertices.max(axis=0)
#     mesh_width = mesh_max[0] - mesh_min[0]
#     mesh_height = mesh_max[1] - mesh_min[1]
#     mesh_depth = mesh_max[2] - mesh_min[2]

#     face_aspect = w / h if h > 0 else 0
#     mesh_aspect = mesh_width / mesh_height if mesh_height > 0 else 0
#     aspect_ratio_diff = abs(face_aspect - mesh_aspect)
    
#     print(f"Face aspect ratio (width/height): {face_aspect:.4f}")
#     print(f"Mesh aspect ratio (width/height): {mesh_aspect:.4f}")
#     print(f"Aspect ratio difference: {aspect_ratio_diff:.4f}")


#     mesh_volume = mesh.volume if hasattr(mesh, 'volume') else 0
#     mesh_area = mesh.area if hasattr(mesh, 'area') else 0
    
#     print(f"Mesh volume: {mesh_volume:.4f}")
#     print(f"Mesh surface area: {mesh_area:.4f}")
#     print(f"Mesh depth (Z-axis): {mesh_depth:.4f}")

#     scale_x = width / mesh_width if mesh_width > 0 else 1
#     scale_y = height / mesh_height if mesh_height > 0 else 1
#     scale = min(scale_x, scale_y) * 0.85 
    
#     simulated_width = int(mesh_width * scale)
#     simulated_height = int(mesh_height * scale)
#     simulated_x = (width - simulated_width) // 2
#     simulated_y = (height - simulated_height) // 2
    
#     mask_rendered = np.zeros((height, width), dtype=np.uint8)
#     cv2.rectangle(mask_rendered, (simulated_x, simulated_y), 
#                   (simulated_x + simulated_width, simulated_y + simulated_height), 255, -1)
#     area_rendered = mask_rendered.sum() / 255.0
    
#     intersection = np.logical_and(mask_original, mask_rendered).sum() / 255.0
#     union = np.logical_or(mask_original, mask_rendered).sum() / 255.0
#     iou = intersection / union if union > 0 else 0
    

#     dice_direct = (2 * intersection) / (area_original + area_rendered) if (area_original + area_rendered) > 0 else 0
    
#     dice_from_iou = (2 * iou) / (1 + iou) if iou > 0 else 0
    
#     print(f"Simulated IoU: {iou:.4f}")
#     print(f"Dice (direct from masks): {dice_direct:.4f}")
#     print(f"Dice (from IoU formula): {dice_from_iou:.4f}")
#     print(f"Note: Dice = 2×IoU/(1+IoU) = {2*iou:.4f}/(1+{iou:.4f}) = {dice_from_iou:.4f}")

#     if mesh_volume > 0 and mesh_area > 0:
#         compactness = (mesh_area ** 3) / (mesh_volume ** 2) if mesh_volume > 0 else 0
#         print(f"  Mesh compactness: {compactness:.2f}")
    
  
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask_original.png"), mask_original)
#     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask_rendered_simulated.png"), mask_rendered)
    
   
#     overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     overlay[mask_original > 0] = [0, 255, 0] 
#     overlay[mask_rendered > 0] = [255, 0, 0] 
#     overlay[(mask_original > 0) & (mask_rendered > 0)] = [255, 255, 0]
#     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


#     results_summary.append({
#         'name': base_name,
#         'face_aspect': face_aspect,
#         'mesh_aspect': mesh_aspect,
#         'aspect_diff': aspect_ratio_diff,
#         'iou': iou,
#         'dice': dice_from_iou,
#         'dice_direct': dice_direct,
#         'volume': mesh_volume,
#         'area': mesh_area,
#         'depth': mesh_depth,
#         'vertices': len(mesh.vertices)
#     })


# print(f"{'Name':<10} {'Face Asp':<10} {'Mesh Asp':<10} {'Diff':<10} {'IoU':<10} {'Dice':<10}")

# for res in results_summary:
#     print(f"{res['name']:<10} {res['face_aspect']:<10.4f} {res['mesh_aspect']:<10.4f} {res['aspect_diff']:<10.4f} {res['iou']:<10.4f} {res['dice']:<10.4f}")


# print(f"{'Name':<10} {'Volume':<12} {'Area':<12} {'Depth':<12} {'Vertices':<12}")

# for res in results_summary:
#     print(f"{res['name']:<10} {res['volume']:<12.4f} {res['area']:<12.4f} {res['depth']:<12.4f} {res['vertices']:<12,}")


# if results_summary:
#     avg_aspect_diff = np.mean([r['aspect_diff'] for r in results_summary])
#     avg_iou = np.mean([r['iou'] for r in results_summary])
#     avg_dice = np.mean([r['dice'] for r in results_summary])
#     avg_volume = np.mean([r['volume'] for r in results_summary])
#     avg_area = np.mean([r['area'] for r in results_summary])
#     avg_vertices = np.mean([r['vertices'] for r in results_summary])

#     print(f"  Average aspect ratio difference: {avg_aspect_diff:.4f}")
#     print(f"  Average IoU: {avg_iou:.4f}")
#     print(f"  Average Dice (from IoU formula): {avg_dice:.4f}")
#     print(f"  Average volume: {avg_volume:.4f}")
#     print(f"  Average surface area: {avg_area:.4f}")
#     print(f"  Average vertex count: {avg_vertices:,.0f}")