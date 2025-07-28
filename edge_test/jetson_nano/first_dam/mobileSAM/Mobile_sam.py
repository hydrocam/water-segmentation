# Final version: MobileSAM + tegrastats logging with segmentation tagging and merged CSV

import os
import cv2
import numpy as np
import torch
import pandas as pd
import time
import threading
import csv
import subprocess
import re
from datetime import datetime
from queue import Queue
from mobile_sam import sam_model_registry, SamPredictor

# Load MobileSAM model
model_type = "vit_t"
checkpoint_path = "/home/uwrl/Model/mobile_sam_vit.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to(device)
sam_model.eval()
predictor = SamPredictor(sam_model)

# Paths
test_image_folder = "/home/uwrl/Model/test_image"
output_folder = "/home/uwrl/Model/segmented_image/mobile_SAM/first_dam"
os.makedirs(output_folder, exist_ok=True)
tegrastats_csv_path = os.path.join(output_folder, "tegrastats_metrics.csv")
final_output_csv = os.path.join(output_folder, "segmentation_metrics_trend.csv")
merged_output_csv = os.path.join(output_folder, "merged_inference_metrics.csv")

# ROIs
trapezium1 = np.array([[810, 670], [980, 620], [1000, 770], [830, 830]], np.int32)
trapezium2 = np.array([[400, 810], [630, 720], [630, 920], [400, 1050]], np.int32)

def perform_segmentation(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    image_rgb = cv2.resize(image_rgb, (640, 640), interpolation=cv2.INTER_AREA)
    predictor.set_image(image_rgb)
    with torch.no_grad():
        masks, _, _ = predictor.predict(point_coords=None, box=None, multimask_output=False)
    mask = masks[0, :, :]
    binary_mask = (mask > 0).astype(np.uint8)
    return cv2.resize(binary_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST), image

def calculate_segmented_pixels(binary_mask, trapezium):
    mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.fillPoly(mask, [trapezium], 1)
    return int(np.sum(binary_mask[mask == 1]))

def tegrastats_logger(csv_path, stop_event):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Timestamp", "RAM_MB", "CPU_Load", "GR3D_FREQ", "EMC_FREQ",
            "Power_Total_mW", "Power_CPU_mW", "Power_GPU_mW",
            "Temp_CPU_C", "Temp_GPU_C"
        ])
        writer.writeheader()

        process = subprocess.Popen(
            ["/usr/bin/tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            if stop_event.is_set():
                break

            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            ram_match = re.search(r"RAM (\d+)/\d+MB", line)
            cpu_match = re.search(r"CPU \[(.*?)\]", line)
            gr3d_match = re.search(r"GR3D_FREQ (\d+%@\d+)", line)
            emc_match = re.search(r"EMC_FREQ (\d+%@\d+)", line)
            power_total_match = re.search(r"POM_5V_IN (\d+)/\d+", line)
            power_cpu_match = re.search(r"POM_5V_CPU (\d+)/\d+", line)
            power_gpu_match = re.search(r"POM_5V_GPU (\d+)/\d+", line)
            temp_cpu_match = re.search(r"CPU@([\d.]+)C", line)
            temp_gpu_match = re.search(r"GPU@([\d.]+)C", line)

            writer.writerow({
                "Timestamp": timestamp,
                "RAM_MB": int(ram_match.group(1)) if ram_match else None,
                "CPU_Load": cpu_match.group(1) if cpu_match else None,
                "GR3D_FREQ": gr3d_match.group(1) if gr3d_match else None,
                "EMC_FREQ": emc_match.group(1) if emc_match else None,
                "Power_Total_mW": int(power_total_match.group(1)) if power_total_match else None,
                "Power_CPU_mW": int(power_cpu_match.group(1)) if power_cpu_match else None,
                "Power_GPU_mW": int(power_gpu_match.group(1)) if power_gpu_match else None,
                "Temp_CPU_C": float(temp_cpu_match.group(1)) if temp_cpu_match else None,
                "Temp_GPU_C": float(temp_gpu_match.group(1)) if temp_gpu_match else None
            })
            f.flush()

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("tegrastats process did not terminate gracefully. Killing it...")
            process.kill()

tegrastats_stop_event = threading.Event()
csv_data = []
tegrastats_thread = threading.Thread(target=tegrastats_logger, args=(tegrastats_csv_path, tegrastats_stop_event))
tegrastats_thread.start()

csv_headers = [
    "Image", "Date", "Time", "Inference_Start",
    "ROI1_pixels", "ROI2_pixels", "Inference_time_sec",
    "Peak_GPU_Mem_MB"
]

with open(final_output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    for image_name in os.listdir(test_image_folder):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_image_folder, image_name)
        print(f"Processing: {image_name}")

        start_time = time.time()
        binary_mask, original_image = perform_segmentation(image_path)
        end_time = time.time()

        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

        roi1 = calculate_segmented_pixels(binary_mask, trapezium1)
        roi2 = calculate_segmented_pixels(binary_mask, trapezium2)

        try:
            timestamp = image_name.split("image_capture_")[1].split(".jpg")[0]
            date, time_part = timestamp.split("_")
        except:
            date, time_part = "Unknown", "Unknown"

        overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlay[binary_mask == 1] = [0, 0, 255]
        cv2.polylines(overlay, [trapezium1], True, (255, 0, 0), 4)
        cv2.polylines(overlay, [trapezium2], True, (255, 0, 0), 4)
        out_path = os.path.join(output_folder, image_name.replace('.jpg', '_segmented.png'))
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        writer.writerow({
            "Image": image_name,
            "Date": date,
            "Time": time_part,
            "Inference_Start": datetime.utcnow(),
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": round(inference_time, 3),
            "Peak_GPU_Mem_MB": round(peak_memory, 2)
        })

        torch.cuda.empty_cache()
        del binary_mask, original_image
        torch.cuda.ipc_collect()

time.sleep(20)  # wait additional seconds for post-inference logging
tegrastats_stop_event.set()
tegrastats_thread.join()

inference_df = pd.read_csv(final_output_csv)

# Merge with tegrastats
metrics_df = pd.read_csv(tegrastats_csv_path)
metrics_df['Timestamp'] = pd.to_datetime(metrics_df['Timestamp'], utc=True)
inference_df['Inference_Start'] = pd.to_datetime(inference_df['Inference_Start'], utc=True)

buffer_sec = 10
inference_windows = list(zip(inference_df['Inference_Start'], inference_df['Inference_time_sec']))

def tag_phase(ts):
    for start, dur in inference_windows:
        if start - pd.Timedelta(seconds=buffer_sec) <= ts <= start + pd.Timedelta(seconds=dur + buffer_sec):
            return "During_Inference"
    return "Idle"

# Build inference time windows with start and end
inference_df['Inference_End'] = inference_df['Inference_Start'] + pd.to_timedelta(inference_df['Inference_time_sec'], unit='s')

# Initialize columns
metrics_df['Inference_Phase'] = 'Idle'
metrics_df['Image'] = ''
metrics_df['ROI1_pixels'] = np.nan
metrics_df['ROI2_pixels'] = np.nan
metrics_df['Inference_time_sec'] = np.nan
metrics_df['Peak_GPU_Mem_MB'] = np.nan

# Tag and enrich
for _, row in inference_df.iterrows():
    mask = (metrics_df['Timestamp'] >= row['Inference_Start'] - pd.Timedelta(seconds=10)) & \
           (metrics_df['Timestamp'] <= row['Inference_End'])
    metrics_df.loc[mask, 'Inference_Phase'] = 'During_Inference'
    metrics_df.loc[mask, 'Image'] = row['Image']
    metrics_df.loc[mask, 'ROI1_pixels'] = row['ROI1_pixels']
    metrics_df.loc[mask, 'ROI2_pixels'] = row['ROI2_pixels']
    metrics_df.loc[mask, 'Inference_time_sec'] = row['Inference_time_sec']
    metrics_df.loc[mask, 'Peak_GPU_Mem_MB'] = row['Peak_GPU_Mem_MB']

metrics_df.to_csv(merged_output_csv, index=False)

print("\nSegmentation CSV saved:", final_output_csv)
print("Merged CSV saved:", merged_output_csv)
print("tegrastats log saved:", tegrastats_csv_path)

