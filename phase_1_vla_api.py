import os, sys
import time
import base64
import uvicorn
from PIL import Image
import numpy as np
from fastapi import FastAPI
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoModelForCausalLM
from prompts import system_prompt_phase_1, user_prompt_template_phase_1
from qwen_vl_utils import process_vision_info
from utils_vlm import ActionTokenizer

device_map = "auto"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
adapter_path = "/mnt/data/songjunru/phase_1_single_initial_64_ckpt/checkpoint-10000/"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
    cache_dir="/home/junru/.cache/models/"
)

processor = AutoProcessor.from_pretrained(model_id, cache_dir="/home/junru/.cache/models/")

# expand the vocab
NUM_BINS = 64   
for i in range(NUM_BINS):
    processor.tokenizer.add_tokens(f"[action_{i}]")
model.resize_token_embeddings(len(processor.tokenizer))

# load the adapter
model.load_adapter(adapter_path)

# load the action tokenizers
action_percentile_1 = np.load("/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/delta_percentile_1.npy")
action_percentile_99 = np.load("/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/delta_percentile_99.npy")
action_tokenizers = []
for i in range(3):
    action_tokenizers.append(ActionTokenizer(bins=NUM_BINS, tokenizer=processor.tokenizer, 
                                             min_action=action_percentile_1[i], max_action=action_percentile_99[i]))

app = FastAPI()

@app.post("/generate")
async def generate(params: dict):
    print("start generating!")
    target_coordinate = params["target_coordinate"]
    current_pose = params["current_pose"]
    image = params["image"]

    image_path = f"{time.time()}.png"
    file = open(image_path,"wb")
    imgdata = base64.b64decode(image[2:-1])
    file.write(imgdata)
    file.close()

    image = Image.open(image_path)

    prompt = user_prompt_template_phase_1.format(current_pose=current_pose, target_coordinate=target_coordinate)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt_phase_1
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    model_inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, temperature=2.0)
    trimmed_generated_ids = [
        out_ids[len(in_ids):].cpu() for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    #output_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    action_tokens_ids = trimmed_generated_ids[0][0:3]
    actions = [action_tokenizers[i].detokenize(action_tokens_ids[i]) for i in range(3)]

    # delete the image
    os.remove(image_path)

    return actions

if __name__ == "__main__":
    uvicorn.run(app, host="172.18.0.1", port=30928)