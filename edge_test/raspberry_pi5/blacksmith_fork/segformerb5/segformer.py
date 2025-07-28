# SegFormer Inference + tegrastats logging with pixel count + system metrics (Updated for Full Compatibility)

import os
import cv2
import numpy as np
import torch
import time
import threading
import csv
from datetime import datetime
from PIL import Image
import pandas as pd
import psutil
import subprocess
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# === Configuration ===
input_dir = "/home/ciroh-uwrl5/CIROH/Model/blacksmith_fork/testing_dataset"
output_dir = "/home/ciroh-uwrl5/CIROH/Model/blacksmith_fork/segformerb5"
os.makedirs(output_dir, exist_ok=True)
log_csv_path = os.path.join(output_dir, "segformer_rpi5_metrics.csv")
seg_csv_path = os.path.join(output_dir, "segformer_segmentation.csv")
model_config = "nvidia/segformer-b5-finetuned-ade-640-640"  # Lighter model for RPi5

target_class = 21  # Class to segment

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_config)
model = SegformerForSemanticSegmentation.from_pretrained(model_config).to(device)
model.eval()

# Trapeziums for ROI
trapezium1 = np.array([[116, 335], [328, 548], [222, 654], [10, 441]], np.int32)
trapezium2 = np.array([[1220, 460], [1480, 340], [1670, 410], [1420, 580]], np.int32)

inference_metadata = []

def get_cpu_temperature():
    temps = psutil.sensors_temperatures()
    for name in temps:
        if 'cpu' in name.lower() or 'soc' in name.lower():
            return temps[name][0].current
    return None

def get_vcgencmd(command):
    try:
        result = subprocess.run(["vcgencmd", command], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except:
        return None

def get_voltage():
    v = get_vcgencmd("measure_volts")
    if v and "volt=" in v:
        return float(v.split("=")[1].replace("V", ""))
    return None

def get_cpu_freq_mhz():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
            return int(f.read().strip()) / 1000
    except:
        return None

def rpi_logger(csv_path, stop_event):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Timestamp", "RAM_MB", "CPU_Usage", "Temp_CPU_C",
            "CPU_Freq_MHz", "Voltage_V", "Estimated_Power_Index",
            "Inference_Phase", "Image", "ROI1_pixels", "ROI2_pixels",
            "Inference_time_sec", "Peak_Mem_MB"
        ])
        writer.writeheader()

        while not stop_event.is_set():
            timestamp = datetime.utcnow()
            ram = psutil.virtual_memory().used / (1024 * 1024)
            cpu = psutil.cpu_percent(interval=1)
            temp = get_cpu_temperature()
            freq = get_cpu_freq_mhz()
            volt = get_voltage()
            power_index = (cpu / 100.0) * (freq / 1000.0) * volt if volt and freq else None

            row = {
                "Timestamp": timestamp,
                "RAM_MB": round(ram, 2),
                "CPU_Usage": round(cpu, 2),
                "Temp_CPU_C": round(temp, 2) if temp else "N/A",
                "CPU_Freq_MHz": round(freq, 2) if freq else "N/A",
                "Voltage_V": round(volt, 3) if volt else "N/A",
                "Estimated_Power_Index": round(power_index, 4) if power_index else "N/A",
                "Inference_Phase": "Idle",
                "Image": "",
                "ROI1_pixels": "",
                "ROI2_pixels": "",
                "Inference_time_sec": "",
                "Peak_Mem_MB": ""
            }

            for meta in inference_metadata:
                if meta["start"] - pd.Timedelta(seconds=5) <= timestamp <= meta["end"] + pd.Timedelta(seconds=5):
                    row.update({
                        "Inference_Phase": "During_Inference",
                        "Image": meta["Image"],
                        "ROI1_pixels": meta["ROI1_pixels"],
                        "ROI2_pixels": meta["ROI2_pixels"],
                        "Inference_time_sec": meta["Inference_time_sec"],
                        "Peak_Mem_MB": meta["Peak_Mem_MB"]
                    })
                    break

            writer.writerow(row)
            f.flush()

# === Start Logger ===
stop_event = threading.Event()
log_thread = threading.Thread(target=rpi_logger, args=(log_csv_path, stop_event))
print("[INFO] Starting system metrics logging...")
log_thread.start()
time.sleep(10)

# === Inference ===
with open(seg_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Image", "Date", "Time", "Inference_Start",
        "ROI1_pixels", "ROI2_pixels", "Inference_time_sec", "Peak_Mem_MB"
    ])
    writer.writeheader()

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, fname)
        image_pil = Image.open(path).convert("RGB")
        image_cv = cv2.imread(path)

        inputs = feature_extractor(images=image_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        start_time = time.time()
        start_dt = pd.to_datetime(datetime.utcnow())

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        end_time = time.time()
        end_dt = pd.to_datetime(datetime.utcnow())
        inf_time = round(end_time - start_time, 3)

        preds = outputs.logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        class_mask = (preds == target_class).astype(np.uint8)
        pred_mask = cv2.resize(class_mask, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = image_cv.copy()
        overlay[pred_mask > 0] = [0, 0, 255]
        cv2.polylines(overlay, [trapezium1], True, (0, 255, 0), 2)
        cv2.polylines(overlay, [trapezium2], True, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, fname.replace('.jpg', '_seg.png')), overlay)

        roi_mask1 = np.zeros_like(pred_mask)
        roi_mask2 = np.zeros_like(pred_mask)
        cv2.fillPoly(roi_mask1, [trapezium1], 1)
        cv2.fillPoly(roi_mask2, [trapezium2], 1)
        roi1 = int(np.sum((pred_mask == 1) & (roi_mask1 == 1)))
        roi2 = int(np.sum((pred_mask == 1) & (roi_mask2 == 1)))

        peak_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        writer.writerow({
            "Image": fname,
            "Date": start_dt.date(),
            "Time": start_dt.time(),
            "Inference_Start": start_dt,
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": inf_time,
            "Peak_Mem_MB": round(peak_mem, 2)
        })

        inference_metadata.append({
            "Image": fname,
            "start": start_dt,
            "end": end_dt,
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": inf_time,
            "Peak_Mem_MB": round(peak_mem, 2)
        })

        print(f"[INFO] Processed {fname} | ROI1: {roi1} | ROI2: {roi2} | Time: {inf_time}s")

# === Cleanup ===
print("[INFO] Waiting 30s post-inference...")
time.sleep(30)
stop_event.set()
log_thread.join()
print("[INFO] Done. All files saved.")
