# Real-Time Industrial Defect Detection & Quality-Control Assistant (Edge AI on Arm)

**Bharat AI-SoC Challenge 2026–27 — Problem Statement 5 (Manufacturing & Industry 4.0)**  
**Platform:** Raspberry Pi 5 (Arm Cortex-A76, 64-bit `aarch64`) | **Runtime:** PyTorch ExecuTorch (`XnnpackBackend`) + ONNX Runtime

---

## 1. Project Overview

This project implements a fully offline, two-stage edge AI quality-control assistant designed for factory floor Printed Circuit Board (PCB) inspection without cloud dependency:

1. **Stage 1 — Real-Time Visual Defect Detection (`YOLOv8n` ONNX):**  
   Detects and localizes six industrial PCB defect classes (`Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`), computes bounding-box severity (`Normal`, `Medium`, `High`), and renders annotated visual overlays.
2. **Stage 2 — On-Device Reasoning & Operator Advisory (`Llama 3.2 1B-Instruct` via ExecuTorch):**  
   Runs an INT8-quantized Llama 3.2 1B model locally on the Arm Cortex-A76 CPU using the PyTorch **ExecuTorch XNNPACK** backend to generate actionable root-cause and corrective instructions for assembly-line operators.
3. **Stage 3 — Edge Telemetry & Local Audit Logging:**  
   Logs every inspection event, vision latency, LLM throughput, peak RSS memory usage, and operator recommendation to `results/inspection_log.csv`.

---

## 2. System Architecture & Pipeline Separation

The vision and language inference pipelines are strictly decoupled to balance real-time optical throughput with low-memory LLM reasoning on a 4 GB edge SoC:

```text
[PCB Optical Frame]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: YOLOv8n Defect Detector (ONNX Runtime / Arm)   │
│  - Input: 640x640 RGB image                              │
│  - Output: Defect class, confidence score, bbox severity │
└───────────────────────────┬──────────────────────────────┘
                            │ (Triggered on defect detection)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: Llama 3.2 1B-Instruct (ExecuTorch + XNNPACK)   │
│  - Format: INT8 quantized .pte static graph [1, 512]     │
│  - Output: Concise operator corrective instruction       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: Local Audit Logger & Annotated Frame Export    │
│  - Saves results/inspection_log.csv & results/yolo_*.jpg │
└──────────────────────────────────────────────────────────┘
