import os
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuration
gt_folder = # INSERT THE PATH TO YOUR GT FOLDER HERE
pred_folder = # INSERT THE PATH TO YOUR PREDICTION FOLDER HERE
IOU_THRESHOLD = # INSERT DESIRED IOU THRES (BELOW THIS THRESHOLD THE PREDICTION WONT BE CONSIDERED IN THE CALCULATION)
IMG_WIDTH = 2592 
IMG_HEIGHT = 1944
PIXEL_TO_MICROMETER = 0.4
HAIR_CLASS_ID = 4

def sort_coordinates(x1, y1, x2, y2, x3, y3, x4, y4):
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    points.sort(key=lambda p: (p[1], p[0]))
    bottom_points = points[:2]
    top_points = points[2:]
    bottom_points.sort(key=lambda p: p[0])
    top_points.sort(key=lambda p: p[0], reverse=True)
    bl, br = bottom_points
    tr, tl = top_points
    return bl[0], bl[1], br[0], br[1], tr[0], tr[1], tl[0], tl[1]

def get_l_and_w(x1, y1, x2, y2, x3, y3, x4, y4):
    x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s = sort_coordinates(x1, y1, x2, y2, x3, y3, x4, y4)
    l = math.sqrt((x2s - x1s)**2 + (y2s - y1s)**2)
    w = math.sqrt((x4s - x1s)**2 + (y4s - y1s)**2)
    return l, w

def calculate_polygon_area(x1, y1, x2, y2, x3, y3, x4, y4):
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    polygon = Polygon(points)
    return polygon.area

def calculate_iou_obb(box1, box2):
    x1, y1, x2, y2, x3, y3, x4, y4 = box1[1:9]
    xa1, ya1, xa2, ya2, xa3, ya3, xa4, ya4 = box2[1:9]
    poly1 = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    poly2 = Polygon([(xa1, ya1), (xa2, ya2), (xa3, ya3), (xa4, ya4)])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union > 0 else 0.0

def parse_annotation_file(filepath):
    annotations = []
    if not os.path.exists(filepath):
        return annotations
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                conf = float(parts[9]) if len(parts) > 9 else 1.0
                annotations.append([class_id] + coords + [conf])
    return annotations

def convert_to_absolute(coords, img_width, img_height):
    abs_coords = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_width
        y = coords[i+1] * img_height
        abs_coords.extend([x, y])
    return abs_coords

def match_predictions_to_gt(gt_boxes, pred_boxes):
    matches = []
    used_gt = set()
    used_pred = set()
    pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][-1], reverse=True)
    for pred_idx in pred_indices:
        if pred_idx in used_pred:
            continue
        pred_box = pred_boxes[pred_idx]
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            iou = calculate_iou_obb(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= IOU_THRESHOLD:
            matches.append((pred_idx, best_gt_idx, best_iou))
            used_pred.add(pred_idx)
            used_gt.add(best_gt_idx)
    return matches, used_pred, used_gt

def calculate_metrics(gt_boxes, pred_boxes):
    metrics = []
    for box in gt_boxes + pred_boxes:
        coords = box[1:9]
        abs_coords = convert_to_absolute(coords, IMG_WIDTH, IMG_HEIGHT)
        l, w = get_l_and_w(*abs_coords)
        area = calculate_polygon_area(*abs_coords)
        l_um = l * PIXEL_TO_MICROMETER
        w_um = w * PIXEL_TO_MICROMETER
        area_um2 = area * (PIXEL_TO_MICROMETER ** 2)
        metrics.append((l_um, w_um, area_um2))
    return metrics

def create_scatter_plot(pred_values, gt_values, xlabel, ylabel, title, version_name):
    if len(pred_values) == 0 or len(gt_values) == 0:
        print(f"No data for {title}")
        return
    
    pred_array = np.array(pred_values).reshape(-1, 1)
    gt_array = np.array(gt_values)
    
    # linear regression
    reg = LinearRegression()
    reg.fit(pred_array, gt_array)
    gt_pred = reg.predict(pred_array)
    r2 = r2_score(gt_array, gt_pred)
    
    # RMSE
    rmse = math.sqrt(np.mean((np.array(pred_values) - np.array(gt_values))**2))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_values, gt_values, alpha=0.6, s=20, color='blue', edgecolors='none')
    
    # regression line
    x_range = np.linspace(min(pred_values), max(pred_values), 100)
    y_regression = reg.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_regression, 'b-', linewidth=2, label='Regression line')
    
    # y=x
    min_val = min(min(pred_values), min(gt_values))
    max_val = max(max(pred_values), max(gt_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1, label='y=x')
    
    # stats
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.05, 0.88, f'y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if 'Density' in title:
        rmse_text = f'RMSE = {rmse:.2f} count/image'
    elif 'Length' in title or 'Width' in title:
        rmse_text = f'RMSE = {rmse:.3f} μm'
    elif 'Area' in title:
        rmse_text = f'RMSE = {rmse:.3f} %'
    else:
        rmse_text = f'RMSE = {rmse:.3f}'
    
    plt.text(0.05, 0.81, rmse_text, transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} ({version_name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
    print(f"{title} ({version_name}): R² = {r2:.3f}, RMSE = {rmse:.3f}, y = {slope:.3f}x + {intercept:.3f}")

def analyze_stomata(include_hair=True):
    version_name = "Including Hair" if include_hair else "Excluding Hair"
    print(f"=== Stomata Analysis - {version_name} (IoU=0.1) ===")
    
    density_data = {'pred': [], 'gt': []}
    length_data = {'pred': [], 'gt': []}
    width_data = {'pred': [], 'gt': []}
    area_data = {'pred': [], 'gt': []}
    
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.txt')]
    
    for filename in gt_files:
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        gt_boxes = parse_annotation_file(gt_path)
        pred_boxes = parse_annotation_file(pred_path)
        
        if not include_hair:
            gt_boxes = [box for box in gt_boxes if box[0] != HAIR_CLASS_ID]
            pred_boxes = [box for box in pred_boxes if box[0] != HAIR_CLASS_ID]
        
        # Density data (count per image)
        density_data['pred'].append(len(pred_boxes))
        density_data['gt'].append(len(gt_boxes))
        
        if not gt_boxes and not pred_boxes:
            continue
        
        all_metrics = calculate_metrics(gt_boxes, pred_boxes)
        gt_metrics = all_metrics[:len(gt_boxes)]
        pred_metrics = all_metrics[len(gt_boxes):]
        matches, used_pred, used_gt = match_predictions_to_gt(gt_boxes, pred_boxes)
        
        for pred_idx, gt_idx, iou in matches:
            pred_l, pred_w, pred_a = pred_metrics[pred_idx]
            gt_l, gt_w, gt_a = gt_metrics[gt_idx]
            
            length_data['pred'].append(pred_l)
            length_data['gt'].append(gt_l)
            width_data['pred'].append(pred_w)
            width_data['gt'].append(gt_w)
            
            total_area = IMG_WIDTH * IMG_HEIGHT * (PIXEL_TO_MICROMETER ** 2)
            area_data['pred'].append((pred_a / total_area) * 100)
            area_data['gt'].append((gt_a / total_area) * 100)
    
    # Create scatter plots with RMSE included
    create_scatter_plot(density_data['pred'], density_data['gt'], 
                       'YOLO Density (count/image)', 'Ground Truth Density (count/image)', 
                       'Stomata Density', version_name)
    create_scatter_plot(length_data['pred'], length_data['gt'], 
                       'YOLO Length (μm)', 'Ground Truth Length (μm)', 
                       'Stomata Length', version_name)
    create_scatter_plot(width_data['pred'], width_data['gt'], 
                       'YOLO Width (μm)', 'Ground Truth Width (μm)', 
                       'Stomata Width', version_name)
    create_scatter_plot(area_data['pred'], area_data['gt'], 
                       'YOLO Area (%)', 'Ground Truth Area (%)', 
                       'Stomata Area', version_name)

if __name__ == "__main__":
    analyze_stomata(include_hair=False)
    analyze_stomata(include_hair=True)