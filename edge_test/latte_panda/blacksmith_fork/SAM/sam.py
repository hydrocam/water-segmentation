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

from segment_anything import SamPredictor, sam_model_registry

# Load SAM model
model_type = "vit_b"
checkpoint_path = "/home/uwrl-panda/Model/blacksmith_fork/SAM/BSF_SAM5122weights_ViTB.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
sam_model = sam_model_registry[model_type](checkpoint=None)
sam_model.load_state_dict(state_dict)
sam_model.to(device)
sam_model.eval()
predictor = SamPredictor(sam_model)

IMAGE_SIZE = (640, 640)

# Paths
test_image_folder = "/home/uwrl-panda/Model/blacksmith_fork/testing_dataset"
output_folder = "/home/uwrl-panda/Model/blacksmith_fork/SAM"
os.makedirs(output_folder, exist_ok=True)
metrics_csv_path = os.path.join(output_folder, "lattepanda_metrics.csv")
final_output_csv = os.path.join(output_folder, "segmentation_metrics_trend.csv")

# ROIs
trapezium1 = np.array([[116, 335], [328, 548], [222, 654], [10, 441]], np.int32)
trapezium2 = np.array([[1220, 460], [1480, 340], [1670, 410], [1420, 580]], np.int32)

# Shared buffer
inference_metadata = []

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

def calculate_segmented_pixels(binary_mask, trapezium):
    mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.fillPoly(mask, [trapezium], 1)
    return int(np.sum(binary_mask[mask == 1]))

def lattepanda_logger(csv_path, stop_event):
    prev_energy = read_rapl_energy_uj()
    prev_time = time.time()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Timestamp", "RAM_MB", "CPU_Usage", "Temp_CPU_C",
            "Energy_uJ_Total", "Instant_Power_mW",
            "Inference_Phase", "Image", "ROI1_pixels", "ROI2_pixels",
            "Inference_time_sec", "Peak_GPU_Mem_MB"
        ])
        writer.writeheader()

        while not stop_event.is_set():
            timestamp = datetime.utcnow()
            ram_mb = psutil.virtual_memory().used / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=1)

            try:
                temp_info = psutil.sensors_temperatures()
                coretemps = temp_info.get('coretemp', [])
                core_vals = [entry.current for entry in coretemps if "Core" in entry.label]
                temp_cpu = round(sum(core_vals) / len(core_vals), 2) if core_vals else None
            except:
                temp_cpu = None

            curr_energy = read_rapl_energy_uj()
            curr_time = time.time()

            if prev_energy is not None and curr_energy is not None:
                delta_energy = curr_energy - prev_energy
                delta_time = curr_time - prev_time
                instant_power_mw = round((delta_energy / delta_time) / 1000, 2) if delta_time > 0 else "N/A"
            else:
                instant_power_mw = "N/A"

            prev_energy = curr_energy
            prev_time = curr_time

            row = {
                "Timestamp": timestamp,
                "RAM_MB": round(ram_mb, 2),
                "CPU_Usage": round(cpu_percent, 2),
                "Temp_CPU_C": round(temp_cpu, 2) if temp_cpu else "N/A",
                "Energy_uJ_Total": curr_energy if curr_energy else "N/A",
                "Instant_Power_mW": instant_power_mw,
                "Inference_Phase": "Idle",
                "Image": "", "ROI1_pixels": "", "ROI2_pixels": "",
                "Inference_time_sec": "", "Peak_GPU_Mem_MB": ""
            }

            for meta in inference_metadata:
                if meta["start"] - pd.Timedelta(seconds=5) <= timestamp <= meta["end"] + pd.Timedelta(seconds=5):
                    row.update({
                        "Inference_Phase": "During_Inference",
                        "Image": meta["Image"],
                        "ROI1_pixels": meta["ROI1_pixels"],
                        "ROI2_pixels": meta["ROI2_pixels"],
                        "Inference_time_sec": meta["Inference_time_sec"],
                        "Peak_GPU_Mem_MB": meta["Peak_GPU_Mem_MB"]
                    })
                    break

            writer.writerow(row)
            f.flush()

def read_rapl_energy_uj():
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None

lattepanda_stop_event = threading.Event()
lattepanda_thread = threading.Thread(target=lattepanda_logger, args=(metrics_csv_path, lattepanda_stop_event))

# Start logger early
print("[INFO] Starting system metrics logging...")
lattepanda_thread.start()
time.sleep(10)  # pre-inference buffer

# Main inference loop
with open(final_output_csv, 'w', newline='') as csvfile:
    fieldnames = ["Image", "Date", "Time", "Inference_Start", "ROI1_pixels", "ROI2_pixels", "Inference_time_sec", "Peak_GPU_Mem_MB"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for image_name in sorted(os.listdir(test_image_folder)):
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

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
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
            "Peak_GPU_Mem_MB": round(peak_memory, 2)
        })

        inference_metadata.append({
            "Image": image_name,
            "start": start_dt,
            "end": end_dt,
            "ROI1_pixels": roi1,
            "ROI2_pixels": roi2,
            "Inference_time_sec": round(inference_time, 3),
            "Peak_GPU_Mem_MB": round(peak_memory, 2)
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        del binary_mask, original_image

print("[INFO] Waiting 30 seconds to capture post-inference system trend...")
time.sleep(30)
lattepanda_stop_event.set()
lattepanda_thread.join()
print("✅ Segmentation CSV saved:", final_output_csv)
print("✅ System metrics CSV saved:", metrics_csv_path)