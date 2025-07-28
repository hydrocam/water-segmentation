# Optimized version: MobileSAM + Optional tegrastats logging + Efficient streaming + ROI tagging

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
from segment_anything import SamPredictor, sam_model_registry

# ====== CONFIGURATION ======
ENABLE_TEGRASTATS = True
SAVE_OUTPUT = True
IMAGE_SIZE = (320, 320)

# ====== PATHS ======
test_image_folder = "/home/uwrl/Model/blacksmith_fork/testing_dataset"
output_folder = "/home/uwrl/Model/blacksmith_fork/SAM"
os.makedirs(output_folder, exist_ok=True)
tegrastats_csv_path = os.path.join(output_folder, "tegrastats_metrics.csv")
final_output_csv = os.path.join(output_folder, "segmentation_metrics_trend.csv")
merged_output_csv = os.path.join(output_folder, "merged_inference_metrics.csv")

# ====== ROIs ======
trapezium1 = np.array([[116, 335], [328, 548], [222, 654], [10, 441]], np.int32)
trapezium2 = np.array([[1220, 460], [1480, 340], [1670, 410], [1420, 580]], np.int32)

# ====== Load SAM Model ======
model_type = "vit_b"
checkpoint_path = "/home/uwrl/Model/first_dam/SAM/FD_SAM5122weights_ViTB.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to(device)
sam_model.eval()
predictor = SamPredictor(sam_model)

# ====== TegraStats Logger ======
def tegrastats_logger(csv_path, stop_event):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Timestamp", "RAM_MB", "CPU_Load", "GR3D_FREQ", "EMC_FREQ",
            "Power_Total_mW", "Power_CPU_mW", "Power_GPU_mW",
            "Temp_CPU_C", "Temp_GPU_C"
        ])
        writer.writeheader()

        try:
            process = subprocess.Popen(
                ["/usr/bin/tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
                bufsize=1
            )
        except Exception as e:
            print(f"❌ Failed to start tegrastats: {e}")
            return

        for line in process.stdout:
            if stop_event.is_set():
                break

            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            ram_match = re.search(r"RAM (\d+)/", line)
            cpu_match = re.search(r"CPU \[(.*?)\]", line)
            gr3d_match = re.search(r"GR3D_FREQ (\d+%)", line)
            emc_match = re.search(r"EMC_FREQ (\d+%)", line)
            power_total_match = re.search(r"VDD_IN (\d+)mW", line)
            power_cpu_match = re.search(r"VDD_CPU_GPU_CV (\d+)mW", line)
            power_gpu_match = re.search(r"VDD_CPU_GPU_CV (\d+)mW", line)
            temp_cpu_match = re.search(r"cpu@([\d.]+)C", line)
            temp_gpu_match = re.search(r"gpu@([\d.]+)C", line)

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

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

# ====== Segmentation and Metrics ======
def perform_segmentation(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    predictor.set_image(resized)
    with torch.no_grad():
        masks, _, _ = predictor.predict(point_coords=None, box=None, multimask_output=False)
    mask = masks[0, :, :]
    binary_mask = (mask > 0).astype(np.uint8)
    return cv2.resize(binary_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST), image

def calculate_segmented_pixels(mask, trapezium):
    roi_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(roi_mask, [trapezium], 1)
    return int(np.sum(mask[roi_mask == 1]))

tegrastats_stop_event = threading.Event()
if ENABLE_TEGRASTATS:
    tegrastats_thread = threading.Thread(target=tegrastats_logger, args=(tegrastats_csv_path, tegrastats_stop_event))
    tegrastats_thread.start()

csv_headers = ["Image", "Date", "Time", "Inference_Start", "ROI1_pixels", "ROI2_pixels", "Inference_time_sec", "Peak_GPU_Mem_MB"]
with open(final_output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    for image_name in sorted(os.listdir(test_image_folder)):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
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

        if SAVE_OUTPUT:
            overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            overlay[binary_mask == 1] = [0, 0, 255]
            cv2.polylines(overlay, [trapezium1], True, (255, 0, 0), 3)
            cv2.polylines(overlay, [trapezium2], True, (255, 0, 0), 3)
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

# Wait for final logging and merge
if ENABLE_TEGRASTATS:
    print("Waiting for tegrastats to finish logging...")
    time.sleep(30)
    tegrastats_stop_event.set()
    tegrastats_thread.join()

inference_df = pd.read_csv(final_output_csv)
if ENABLE_TEGRASTATS:
    metrics_df = pd.read_csv(tegrastats_csv_path)
    metrics_df['Timestamp'] = pd.to_datetime(metrics_df['Timestamp'], utc=True)
    inference_df['Inference_Start'] = pd.to_datetime(inference_df['Inference_Start'], utc=True)
    inference_df['Inference_End'] = inference_df['Inference_Start'] + pd.to_timedelta(inference_df['Inference_time_sec'], unit='s')

    metrics_df['Inference_Phase'] = 'Idle'
    metrics_df['Image'] = ''
    metrics_df['ROI1_pixels'] = np.nan
    metrics_df['ROI2_pixels'] = np.nan
    metrics_df['Inference_time_sec'] = np.nan
    metrics_df['Peak_GPU_Mem_MB'] = np.nan

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

print("\n✅ Segmentation CSV saved:", final_output_csv)
if ENABLE_TEGRASTATS:
    print("✅ Merged CSV saved:", merged_output_csv)
    print("✅ tegrastats log saved:", tegrastats_csv_path)

