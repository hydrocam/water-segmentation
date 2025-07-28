import os
import cv2
import numpy as np
import torch
import pandas as pd
import time
import threading
import csv
from datetime import datetime
import psutil
import subprocess
from segment_anything import SamPredictor, sam_model_registry

# ====== CONFIGURATION ======
IMAGE_SIZE = (320, 320)

# Paths
test_image_folder = "/home/ciroh-uwrl5/CIROH/Model/first_dam/testing_dataset"
output_folder = "/home/ciroh-uwrl5/CIROH/Model/first_dam/SAM"
os.makedirs(output_folder, exist_ok=True)
metrics_csv_path = os.path.join(output_folder, "rpi5_metrics.csv")
final_output_csv = os.path.join(output_folder, "segmentation_metrics_trend.csv")

# ====== ROIs ======
trapezium1 = np.array([[810, 670], [980, 620], [1000, 770], [830, 830]], np.int32)
trapezium2 = np.array([[400, 810], [630, 720], [630, 920], [400, 1050]], np.int32)

# ====== Load SAM Model ======
model_type = "vit_b"
checkpoint_path = "/home/ciroh-uwrl5/CIROH/Model/first_dam/SAM/FD_SAM5122weights_ViTB.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
sam_model = sam_model_registry[model_type](checkpoint=None)
sam_model.load_state_dict(state_dict)
sam_model.to(device)
sam_model.eval()
predictor = SamPredictor(sam_model)

# Shared buffer
inference_metadata = []

def perform_segmentation(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
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

def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        for name in temps:
            if 'cpu' in name.lower() or 'soc' in name.lower():
                return temps[name][0].current
    except:
        pass
    return None

def get_vcgencmd(command):
    try:
        result = subprocess.run(["vcgencmd", command], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except:
        return None

def get_voltage():
    output = get_vcgencmd("measure_volts")
    if output and "volt=" in output:
        return float(output.split('=')[1].replace('V', ''))
    return None

def get_cpu_freq_mhz():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
            return int(f.read().strip()) / 1000  # MHz
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
            ram_mb = psutil.virtual_memory().used / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=1)
            temp_cpu = get_cpu_temperature()
            cpu_freq = get_cpu_freq_mhz()
            voltage = get_voltage()

            if voltage and cpu_freq:
                power_index = (cpu_percent / 100.0) * (cpu_freq / 1000.0) * voltage
            else:
                power_index = None

            row = {
                "Timestamp": timestamp,
                "RAM_MB": round(ram_mb, 2),
                "CPU_Usage": round(cpu_percent, 2),
                "Temp_CPU_C": round(temp_cpu, 2) if temp_cpu else "N/A",
                "CPU_Freq_MHz": round(cpu_freq, 2) if cpu_freq else "N/A",
                "Voltage_V": round(voltage, 3) if voltage else "N/A",
                "Estimated_Power_Index": round(power_index, 4) if power_index else "N/A",
                "Inference_Phase": "Idle",
                "Image": "",
                "ROI1_pixels": "",
                "ROI2_pixels": "",
                "Inference_time_sec": "",
                "Peak_Mem_MB": ""
            }

            for meta in inference_metadata:
                start = meta["start"]
                end = meta["end"]
                if start - pd.Timedelta(seconds=5) <= timestamp <= end + pd.Timedelta(seconds=5):
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

rpi_stop_event = threading.Event()
rpi_thread = threading.Thread(target=rpi_logger, args=(metrics_csv_path, rpi_stop_event))

# Start logger
print("[INFO] Starting system metrics logging...")
rpi_thread.start()
time.sleep(10)

# Main inference loop
with open(final_output_csv, 'w', newline='') as csvfile:
    fieldnames = [
        "Image", "Date", "Time", "Inference_Start",
        "ROI1_pixels", "ROI2_pixels", "Inference_time_sec", "Peak_Mem_MB"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for image_name in os.listdir(test_image_folder):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_image_folder, image_name)
        print(f"[INFO] Processing: {image_name}")

        start_time = time.time()
        start_dt = pd.to_datetime(datetime.utcnow())
        binary_mask, original_image = perform_segmentation(image_path)
        end_time = time.time()
        end_dt = pd.to_datetime(datetime.utcnow())

        inference_time = end_time - start_time
        process = psutil.Process(os.getpid())
        peak_memory = process.memory_info().rss / (1024 * 1024)

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
            "Inference_Start": start_dt,
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": round(inference_time, 3),
            "Peak_Mem_MB": round(peak_memory, 2)
        })

        inference_metadata.append({
            "Image": image_name,
            "start": start_dt,
            "end": end_dt,
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": round(inference_time, 3),
            "Peak_Mem_MB": round(peak_memory, 2)
        })

        del binary_mask, original_image

print("[INFO] Waiting 30 seconds to capture post-inference trend...")
time.sleep(30)
rpi_stop_event.set()
rpi_thread.join()

print("\n[INFO] Segmentation CSV saved:", final_output_csv)
print("[INFO] Metrics CSV saved:", metrics_csv_path)
