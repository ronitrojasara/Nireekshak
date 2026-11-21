import os
import csv
import time
import psutil
import cv2
import torch
from datetime import datetime
from ultralytics import YOLO
from tokenizers import Tokenizer
from executorch.runtime import Runtime

# ---------------------------------------------------------
# STAGE 1: Real YOLOv8n ONNX Defect Detector on Arm CPU
# ---------------------------------------------------------
class YOLODefectDetector:
    def __init__(self, model_path="./vision_model/pcb_yolov8n.onnx"):
        print("[Init] Loading YOLOv8n ONNX PCB Defect Detector...")
        self.model = YOLO(model_path, task="detect")

    def inspect(self, image_path, conf_thresh=0.35):
        t0 = time.perf_counter()
        results = self.model.predict(source=image_path, imgsz=640, conf=conf_thresh, verbose=False)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        res = results[0]
        annotated = res.plot()  # Draws bounding boxes, class labels & confidence scores

        if len(res.boxes) == 0:
            return {
                "defect": "None",
                "severity": "Normal",
                "confidence": 0.99,
                "vision_ms": latency_ms,
                "annotated": annotated
            }

        # Pick highest-confidence defect detection
        best_idx = int(torch.argmax(res.boxes.conf).item())
        cls_id = int(res.boxes.cls[best_idx].item())
        defect_name = res.names[cls_id]
        conf = round(float(res.boxes.conf[best_idx].item()), 2)

        # Calculate severity from normalized bounding-box area
        xywhn = res.boxes.xywhn[best_idx]
        area_pct = float(xywhn[2] * xywhn[3]) * 100
        if defect_name.lower() in ["open_circuit", "short"] or area_pct > 1.5:
            severity = "High"
        else:
            severity = "Medium"

        return {
            "defect": defect_name,
            "severity": severity,
            "confidence": conf,
            "vision_ms": latency_ms,
            "annotated": annotated
        }

# ---------------------------------------------------------
# STAGE 2: Mandatory Llama 3.2 1B Advisory via ExecuTorch
# ---------------------------------------------------------
class ExecuTorchLlamaAdvisor:
    def __init__(self, model_dir="./llama_model"):
        print("[Init] Loading Llama 3.2 1B INT8 (.pte) on Arm XNNPACK...")
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.rt = Runtime.get()
        self.prog = self.rt.load_program(os.path.join(model_dir, "model.pte"))
        self.method = self.prog.load_method("forward")
        self.eos_id = self.tokenizer.token_to_id("<|eot_id|>")
        self.pad_id = self.tokenizer.token_to_id("<|finetune_right_pad_id|>") or 128004

    def advise(self, defect_type, severity, confidence, max_new_tokens=16):
        if defect_type == "None":
            return "Board passed optical inspection. Proceed to next assembly stage.", 0.0, 0

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a factory QC assistant. Give a concise 1-sentence operator action.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Defect={defect_type}, Severity={severity}, Confidence={int(confidence*100)}%. Fix?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        MAX_SEQ = 512
        generated = []
        t0 = time.perf_counter()

        for _ in range(max_new_tokens):
            cur_len = len(tokens)
            if cur_len >= MAX_SEQ:
                break
            padded = tokens + [self.pad_id] * (MAX_SEQ - cur_len)
            mask = [1] * cur_len + [0] * (MAX_SEQ - cur_len)

            out = self.method.execute([
                torch.tensor([padded], dtype=torch.int64),
                torch.tensor([mask], dtype=torch.int64)
            ])[0]
            next_tok = int(torch.argmax(out[0, cur_len - 1]).item())
            if next_tok == self.eos_id:
                break
            tokens.append(next_tok)
            generated.append(next_tok)

        elapsed = time.perf_counter() - t0
        tps = round(len(generated) / max(elapsed, 0.001), 3)
        return self.tokenizer.decode(generated).strip(), tps, len(generated)

# ---------------------------------------------------------
# STAGE 3: End-to-End Edge Pipeline & Local CSV Logger
# ---------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)
    process = psutil.Process(os.getpid())
    detector = YOLODefectDetector()
    advisor = ExecuTorchLlamaAdvisor()

    log_path = "results/inspection_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "sample_file", "defect_type", "severity",
            "confidence", "vision_latency_ms", "llm_tokens",
            "llm_tok_per_sec", "peak_ram_mb", "operator_recommendation"
        ])

    img_dir = "./vision_model/test_images"
    test_files = sorted([os.path.join(img_dir, s) for s in os.listdir(img_dir) if s.lower().endswith((".jpg", ".png"))])[:3]

    print("\n" + "="*68)
    print("Nireekshak: YOLOv8n + EXECUTORCH LLAMA 3.2 QC ASSISTANT")
    print("="*68)

    for path in test_files:
        res = detector.inspect(path)
        cv2.imwrite(os.path.join("results", "yolo_" + os.path.basename(path)), res["annotated"])

        print(f"\n[Inspecting] {os.path.basename(path)}")
        print(f"  -> YOLOv8n Vision: {res['defect']} | Severity: {res['severity']} | Conf: {res['confidence']*100:.0f}% ({res['vision_ms']} ms)")

        advice, tps, n_tok = advisor.advise(res["defect"], res["severity"], res["confidence"], max_new_tokens=16)
        ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)

        print(f"  -> Llama Advisory: {advice}")
        print(f"  -> Edge Metrics  : {n_tok} tokens @ {tps} tok/s | RSS RAM: {ram_mb} MB")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                os.path.basename(path),
                res["defect"],
                res["severity"],
                res["confidence"],
                res["vision_ms"],
                n_tok,
                tps,
                ram_mb,
                advice
            ])

    print("\n[Done] Results saved to results/inspection_log.csv and results/yolo_*.jpg")

if __name__ == "__main__":
    main()
