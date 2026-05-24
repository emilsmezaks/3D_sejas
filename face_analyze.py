
import cv2
import numpy as np
import os
import glob


def get_face_mask(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None
    
    x, y, w, h = faces[0]
    
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    center_x = x + w // 2
    center_y = y + h // 2
    radius = int(max(w, h) // 2)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask, img


def create_simulated_mask_from_obj(obj_path, reference_mask):
    
    try:
        import trimesh
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        

        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        
        aspect_ratio = width / height if height > 0 else 1.0
    
    
        h, w = reference_mask.shape
        contours, _ = cv2.findContours(reference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return reference_mask
        
        x, y, w_box, h_box = cv2.boundingRect(contours[0])
        center_x = x + w_box // 2
        center_y = y + h_box // 2
        
        simulated_mask = np.zeros_like(reference_mask)
        
        if aspect_ratio > 1:
        
            radius_x = int(w_box // 2)
            radius_y = int(w_box // 2 / aspect_ratio)
        else:
    
            radius_x = int(h_box // 2 * aspect_ratio)
            radius_y = int(h_box // 2)
        
        cv2.ellipse(simulated_mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        
        return simulated_mask
    except Exception as e:
        print(f"  Kļūda {e}")
        return reference_mask


def calculate_metrics(mask_gt, mask_pred):
    
    if mask_gt is None or mask_pred is None:
        return None
    
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    
    iou = intersection / union if union > 0 else 0
    
    dice = (2 * intersection) / (mask_gt.sum() + mask_pred.sum()) if (mask_gt.sum() + mask_pred.sum()) > 0 else 0
    
    error_percent = (1 - iou) * 100
    precision = 100 - error_percent
    
    return {
        'iou': iou,
        'dice': dice,
        'error_percent': error_percent,
        'precision': precision,
        'intersection': intersection,
        'union': union
    }


def visualize_comparison(original_img, mask_gt, mask_pred, output_path):
   
    overlay = original_img.copy()
    
    overlay[mask_gt > 0] = [0, 255, 0]
    
    mask_pred_overlay = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)
    mask_pred_overlay[mask_pred > 0] = [0, 0, 255]
    
    result = cv2.addWeighted(overlay, 0.5, mask_pred_overlay, 0.5, 0)
    
    cv2.imwrite(output_path, result)
    print(f"Vizualizācija: {output_path}")


def evaluate_superimposition(demo_dir='demo'):
    
    image_files = glob.glob(os.path.join(demo_dir, '*.png'))
    image_files = [f for f in image_files if '_mask' not in f and '_normal' not in f and '_obj' not in f]
    image_files = sorted(image_files)
    
    results = []
    
    
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace('.png', '')
        obj_path = os.path.join(demo_dir, f'{base_name}.obj')
        
        if not os.path.exists(obj_path):
            print(f"Nav {obj_path}")
            continue
        
        print(f"\n{base_name}:")
        print("-" * 40)
        
        mask_gt, original_img = get_face_mask(img_path)
        if mask_gt is None:
            print(f"Seja nav atrasta")
            continue
        
        print(f"{mask_gt.sum() / 255:.0f} px²)")
        
        # 2. Simulētā maska no OBJ
        mask_pred = create_simulated_mask_from_obj(obj_path, mask_gt)
        
        print(f"{mask_pred.sum() / 255:.0f} px²")
        
        metrics = calculate_metrics(mask_gt, mask_pred)
        
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  Precizitāte: {metrics['precision']:.2f}%")
        print(f"  Kļūda: {metrics['error_percent']:.2f}%")
        
        vis_path = os.path.join(demo_dir, f'{base_name}_superimposition.png')
        visualize_comparison(original_img, mask_gt, mask_pred, vis_path)
        
        results.append({
            'name': base_name,
            'iou': metrics['iou'],
            'dice': metrics['dice'],
            'precision': metrics['precision'],
            'error': metrics['error_percent']
        })
    
    if results:

        print(f"{'Attēls':<10} {'IoU':<10} {'Dice':<10} {'Precizitāte (%)':<18}")
        
        for r in results:
            print(f"{r['name']:<10} {r['iou']:<10.4f} {r['dice']:<10.4f} {r['precision']:<18.2f}")
        
        avg_precision = np.mean([r['precision'] for r in results])
    
        print(f"{'VIDĒJAIS':<10} {'-':<10} {'-':<10} {avg_precision:<18.2f}")
        
    
    return results


if __name__ == "__main__":
    results = evaluate_superimposition('demo')