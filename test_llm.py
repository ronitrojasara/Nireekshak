import time
import torch
from tokenizers import Tokenizer
from executorch.runtime import Runtime

print("Loading tokenizer and ExecuTorch model...")
t0 = time.time()
tokenizer = Tokenizer.from_file("./llama_model/tokenizer.json")
rt = Runtime.get()
prog = rt.load_program("./llama_model/model.pte")
method = prog.load_method("forward")
print(f"Loaded in {time.time() - t0:.2f}s")

prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a factory quality-control AI. Give a brief 1-sentence root cause and immediate fix.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Detected defect: PCB Open_Circuit, Severity: High, Confidence: 94%. Action?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
eos_id = tokenizer.token_to_id("<|eot_id|>")
pad_id = tokenizer.token_to_id("<|finetune_right_pad_id|>") or 128004

MAX_SEQ = 512
MAX_NEW_TOKENS = 24

print("Generating operator advisory...")
t_gen = time.time()
generated_ids = []

for step in range(MAX_NEW_TOKENS):
    cur_len = len(tokens)
    if cur_len >= MAX_SEQ:
        break
    # Pad sequence to fixed [1, 512] shape required by model.pte
    padded_ids = tokens + [pad_id] * (MAX_SEQ - cur_len)
    attn_mask = [1] * cur_len + [0] * (MAX_SEQ - cur_len)

    input_ids = torch.tensor([padded_ids], dtype=torch.int64)
    mask_tensor = torch.tensor([attn_mask], dtype=torch.int64)

    outputs = method.execute([input_ids, mask_tensor])
    logits = outputs[0]  # shape [1, 512, 128256]
    next_token = int(torch.argmax(logits[0, cur_len - 1]).item())

    if next_token == eos_id:
        break
    tokens.append(next_token)
    generated_ids.append(next_token)

elapsed = time.time() - t_gen
reply = tokenizer.decode(generated_ids)
print(f"\n--- Llama 3.2 Output ({len(generated_ids)} tokens in {elapsed:.2f}s | {len(generated_ids)/elapsed:.2f} tok/s) ---")
print(reply)
