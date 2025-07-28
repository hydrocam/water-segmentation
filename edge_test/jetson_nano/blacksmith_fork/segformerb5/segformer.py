# SegFormer Inference + tegrastats logging with pixel count + system metrics (Updated for Full Compatibility)

import os
import cv2
import numpy as np
import torch
import time
import threading
import csv
import subprocess
import re
from datetime import datetime
from PIL import Image
import pandas as pd
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Configuration
folder_path = "/home/uwrl/Model/blacksmith_fork/testing_dataset"
model_config = "nvidia/segformer-b5-finetuned-ade-640-640"
output_dir = "/home/uwrl/Model/blacksmith_fork/segformerb5"
csv_file_path = "/home/uwrl/Model/blacksmith_fork/segformerb5/segmentation_segformer.csv"
tegrastats_csv_path = os.path.join(output_dir, "tegrastats_metrics.csv")
merged_csv_path = os.path.join(output_dir, "merged_inference_metrics.csv")
target_class = 21

os.makedirs(output_dir, exist_ok=True)

# Load model and extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_config)
model = SegformerForSemanticSegmentation.from_pretrained(model_config)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

def tegrastats_logger(path, stop_event):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Timestamp", "RAM_MB", "CPU_Load", "GR3D_FREQ", "EMC_FREQ",
            "Power_Total_mW", "Power_CPU_mW", "Power_GPU_mW",
            "Temp_CPU_C", "Temp_GPU_C"
        ])
        writer.writeheader()

        process = subprocess.Popen([
            "sudo", "tegrastats", "--interval", "500"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)

        for line in process.stdout:
            if stop_event.is_set():
                break

            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            ram = re.search(r"RAM (\d+)/", line)
            cpu = re.search(r"CPU \[(.*?)\]", line)
            gr3d = re.search(r"GR3D_FREQ (\d+%@\d+)", line)
            emc = re.search(r"EMC_FREQ (\d+%@\d+)", line)
            pwr_in = re.search(r"POM_5V_IN (\d+)/", line)
            pwr_cpu = re.search(r"POM_5V_CPU (\d+)/", line)
            pwr_gpu = re.search(r"POM_5V_GPU (\d+)/", line)
            temp_cpu = re.search(r"CPU@([\d.]+)C", line)
            temp_gpu = re.search(r"GPU@([\d.]+)C", line)

            writer.writerow({
                "Timestamp": timestamp,
                "RAM_MB": int(ram.group(1)) if ram else None,
                "CPU_Load": cpu.group(1) if cpu else None,
                "GR3D_FREQ": gr3d.group(1) if gr3d else None,
                "EMC_FREQ": emc.group(1) if emc else None,
                "Power_Total_mW": int(pwr_in.group(1)) if pwr_in else None,
                "Power_CPU_mW": int(pwr_cpu.group(1)) if pwr_cpu else None,
                "Power_GPU_mW": int(pwr_gpu.group(1)) if pwr_gpu else None,
                "Temp_CPU_C": float(temp_cpu.group(1)) if temp_cpu else None,
                "Temp_GPU_C": float(temp_gpu.group(1)) if temp_gpu else None
            })
            f.flush()

        process.terminate()

# Trapeziums for ROI
trapezium1 = np.array([[116, 335], [328, 548], [222, 654], [10, 441]], np.int32)
trapezium2 = np.array([[1220, 460], [1480, 340], [1670, 410], [1420, 580]], np.int32)

# Start tegrastats logger
stop_event = threading.Event()
tegra_thread = threading.Thread(target=tegrastats_logger, args=(tegrastats_csv_path, stop_event))
tegra_thread.start()

# Begin inference
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Inference_Start', 'Inference_End', 'ROI1_pixels', 'ROI2_pixels', 'Inference_time_sec'])

    for image_filename in sorted(os.listdir(folder_path)):
        if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_filename)
            image_pil = Image.open(image_path).convert("RGB")
            image = cv2.imread(image_path)

            inputs = feature_extractor(images=image_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(model.device)

            inference_start = datetime.utcnow()
            start_time = time.time()

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            end_time = time.time()
            inference_end = datetime.utcnow()
            inference_duration = round(end_time - start_time, 3)

            predictions = outputs.logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            class_mask = (predictions == target_class).astype(np.uint8)
            pred_mask = cv2.resize(class_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            overlay = image.copy()
            overlay[pred_mask > 0] = [0, 0, 255]
            cv2.polylines(overlay, [trapezium1], True, (0, 255, 0), 2)
            cv2.polylines(overlay, [trapezium2], True, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, image_filename.replace('.jpg', '_seg.png')), overlay)

            roi_mask1 = np.zeros_like(pred_mask)
            roi_mask2 = np.zeros_like(pred_mask)
            cv2.fillPoly(roi_mask1, [trapezium1], 1)
            cv2.fillPoly(roi_mask2, [trapezium2], 1)
            roi1_count = int(np.sum((pred_mask == 1) & (roi_mask1 == 1)))
            roi2_count = int(np.sum((pred_mask == 1) & (roi_mask2 == 1)))

            writer.writerow([image_filename, inference_start, inference_end, roi1_count, roi2_count, inference_duration])
            print(f"Processed {image_filename} | ROI1: {roi1_count}, ROI2: {roi2_count}, Time: {inference_duration}s")

# Graceful shutdown
print("Waiting for post-inference logging...")
time.sleep(30)
stop_event.set()
tegra_thread.join()
print("✅ All metrics and segmentation saved.")

# === Merge segmentation CSV with tegrastats ===
inference_df = pd.read_csv(csv_file_path)
metrics_df = pd.read_csv(tegrastats_csv_path)

metrics_df['Timestamp'] = pd.to_datetime(metrics_df['Timestamp'], utc=True)
inference_df['Inference_Start'] = pd.to_datetime(inference_df['Inference_Start'], utc=True)
inference_df['Inference_End'] = pd.to_datetime(inference_df['Inference_End'], utc=True)

metrics_df['Inference_Phase'] = 'Idle'
metrics_df['ROI1_pixels'] = np.nan
metrics_df['ROI2_pixels'] = np.nan
metrics_df['Inference_time_sec'] = np.nan
metrics_df['Image'] = ''

for idx, row in inference_df.iterrows():
    mask = (metrics_df['Timestamp'] >= row['Inference_Start']) & (metrics_df['Timestamp'] <= row['Inference_End'])
    metrics_df.loc[mask, 'Inference_Phase'] = 'During_Inference'
    metrics_df.loc[mask, 'ROI1_pixels'] = row['ROI1_pixels']
    metrics_df.loc[mask, 'ROI2_pixels'] = row['ROI2_pixels']
    metrics_df.loc[mask, 'Inference_time_sec'] = row['Inference_time_sec']
    metrics_df.loc[mask, 'Image'] = row.get('Image', f"image_{idx+1:03d}")

metrics_df.to_csv(merged_csv_path, index=False)
print("✅ Merged CSV saved:", merged_csv_path)

