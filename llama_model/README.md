---
license: llama3.2
pipeline_tag: text-generation
library_name: executorch
base_model: meta-llama/Llama-3.2-1B-Instruct
base_model_relation: quantized
tags:
- arm
- edge-linux
- executorch
- xnnpack
- int8
- hellaswag
---

# Llama-3.2-1B-Instruct optimized for Arm-based Edge Linux

Llama-3.2-1B-Instruct text generation model, optimized with INT8 quantization and exported as an ExecuTorch program for the XNNPACK CPU backend on Arm-based platforms.

This model was optimized using [Arm AI Model Optimizer](https://developer.arm.com/dev-docs/ai-model-optimizer/).

## Summary

This repository contains an optimized version of meta-llama/Llama-3.2-1B-Instruct for text generation. The model is provided as an ExecuTorch `.pte` file using the XNNPACK delegate, targeting Edge Linux systems. Precision: INT8 weights (quantization details not recorded in metadata).

**Key results**

| Area | Result |
|---|---|
| Model format | ExecuTorch (.pte) |
| Target device class | Edge Linux |
| Reference device | Raspberry Pi 5 (16GB) (Cortex-A76, 4 cores) |
| Primary performance result | 374.72 ms p50 end-to-end latency |
| Accuracy result | HellaSwag accuracy 61.00% (-1.00 pp vs. baseline) |
| Size / memory result | 1.70x smaller model (2,995.74 MB to 1,763.71 MB), 2,074.73 MB peak memory |

## Original model

| Field | Value |
|---|---|
| Original model | meta-llama/Llama-3.2-1B-Instruct |
| Original source | Hugging Face |
| Original developer | Meta |
| Original model card | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Original license | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) |

## Model files

| File | Description |
|---|---|
| `model_prefill.pte` | Optimized INT8 ExecuTorch program (prefill) for deployment |
| `model_decode.pte` | Optimized INT8 ExecuTorch program (decode) for deployment |
| `metadata.yaml` | Optimized model metadata |
| `benchmarks/` | Baseline and optimized benchmark records |

## Performance

Performance was measured on the reference configuration below. Results do not guarantee identical performance on every Arm-based system.

**Reference configuration**

| Field | Value |
|---|---|
| Device / platform | Raspberry Pi 5 (16GB) |
| CPU / accelerator | Cortex-A76, 4 cores, 2.4 GHz |
| OS | Raspberry Pi OS Bookworm |
| Runtime | ExecuTorch 1.1.0 |
| Backend / delegate | XNNPACK |
| Batch size | 1 |
| Prompt length | 512 |
| Generation length | 1 |
| Precision | INT8 weights (quantization details not recorded in metadata) |
| Runs | 30 measured |

**Performance results**

| Metric | Original / baseline | Optimized | Improvement |
|---|---:|---:|---:|
| p50 end-to-end latency | - | 374.72 ms | - |
| Prefill time | - | 5,372.26 ms | - |
| Decode throughput | - | 2.67 tokens/sec | - |
| Peak memory | - | 2,074.73 MB | - |
| Model size | 2,995.74 MB | 1,763.71 MB | 1.70x smaller |

## Accuracy

**Evaluation setup**

| Field | Value |
|---|---|
| Benchmark | HellaSwag |
| Evaluation runtime | ExecuTorch |

**Accuracy results**

| Metric | Original / baseline | Optimized | Change |
|---|---:|---:|---:|
| HellaSwag accuracy | 62.00% | 61.00% | -1.00 pp |

Users should re-evaluate the model on their own data before production use.

## Optimization approach

| Optimization area | Applied? | Notes |
|---|---|---|
| Model conversion | Yes | PyTorch model exported to ExecuTorch (.pte) |
| Quantization | Yes | INT8 weights (quantization details not recorded in metadata) |
| Runtime/backend selection | Yes | XNNPACK |
| Accuracy validation | Yes | HellaSwag |
| Performance validation | Yes | Measured on Raspberry Pi 5 (16GB) |

## Intended use

This model is intended for developers evaluating text generation workloads on Arm-based Edge Linux platforms. It is suitable as a reference implementation for benchmarking, prototyping, and integration exploration.

## Limitations

- Performance depends on the target device, runtime version, backend support, memory configuration, and system load.
- Quality was measured on HellaSwag only. Task-level behavior on other benchmarks was not evaluated.
- This repository is not a replacement for the original model documentation.

## About this version

Original Model:
meta-llama/Llama-3.2-1B-Instruct by Meta - [Repository](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

Optimization/conversion:
INT8 quantization and ExecuTorch `.pte` export with XNNPACK backend support for execution on Arm-based platforms.

Converted/optimized by:
Nota AI, using Arm AI Model Optimizer

License:
The Original Model and the Optimized Model are subject to [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).

This repository contains a converted or optimized version of the Original Model (the "Optimized Model"). The Original Model has been converted or optimized as described above for execution on Arm-based platforms.

No retraining or fine-tuning of the Original Model was performed as part of the conversion or optimization. The conversion or optimization was not intended to change the Original Model's behavior or intended use.

### Original Model and Documentation

For information about the Original Model, including its development, training data, intended uses, limitations and other relevant information, please refer to the [Original Model repository](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Information in that repository was provided by the original developer or other third parties and, unless expressly stated otherwise, has not been independently verified by Arm.

### Licenses and Third-Party Terms

Use of the Original Model and the Optimized Model is subject to the applicable licenses, usage restrictions and other terms identified above and in the relevant repositories. Publication of the Optimized Model does not grant any rights beyond those provided under the applicable license terms.

You are responsible for reviewing those terms and ensuring that your use of the Original Model and the Optimized Model is permitted.

### Purpose of this Release

The Optimized Model is provided as a reference implementation to demonstrate and evaluate execution and performance on Arm-based systems. It is not a production-ready or supported solution.

Arm's publication of the Optimized Model does not constitute an endorsement or certification of the Original Model or a representation that the Optimized Model is suitable for production use or any particular purpose.

To the fullest extent permitted by applicable law (i) the Optimized Model is provided "as is." Arm makes no representations or warranties that the Original Model, the Optimized Model or their outputs are accurate, safe, secure, non-infringing, legally compliant, suitable for production use or fit for any particular purpose; and (ii) Arm will not be liable for any loss or damage arising from or in connection with the Optimized Model, its use or its outputs.

You are responsible for independently evaluating the Optimized Model, its outputs and its suitability for your intended use, including compliance with applicable legal, regulatory, safety and security requirements.

Arm does not commit to provide ongoing support, maintenance or updates for the Optimized Model. Any use of or reliance on the Optimized Model or its outputs is at your own risk.
