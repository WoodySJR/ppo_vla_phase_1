import os, sys, torch, json, random
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import numpy as np
from utils_vlm import format_data_ppo, collate_fn, ActionTokenizer, generate_text_from_sample, generate_actions_from_sample
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# import wandb
import swanlab as wandb
from trl import SFTTrainer
from functools import partial
import pickle
from accelerate import PartialState
from concurrent.futures import ThreadPoolExecutor, as_completed
from peft import get_peft_model
from tqdm import tqdm

from prompts import system_prompt_uav_flow
# BitsAndBytesConfig int-4 config
from configs.bitsandbytes_config import bnb_config
# Configure LoRA
from configs.peft_config import peft_config
# Configure training arguments
from configs.training_config import training_args

# using DDP (Distributed Data Parallel)
# device_string = PartialState().process_index
# device_map = {'': device_string}
# device_map = "auto"
device_map = "balanced"

# set random seed
seed = 928
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# load in the processor (including tokenizer and image processor)
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, cache_dir="/home/junru/.cache/models")

# print(processor.tokenizer.encode("<|im_start|>assistant\n"))
# print(processor.tokenizer.encode("<|im_end|>\n"))

# define a collate function for the trainer
collate_fn_for_trainer = partial(
    collate_fn,
    processor=processor,
)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
    cache_dir="/home/junru/.cache/models"
)

# initialize wandb for logging (need to input token when logging for the first time)
# os.environ["WANDB_BASE_URL"] = "https://api.wandb-cn.top"
wandb.init(
    project="qwen2.5-VL-7b-instruct-phase-1-single-initial",  # change this
    name=f"1step_delta_64_2e-5_0.3_32_alllinear_0910",  # change this
    config=training_args,
)

# add 256 action tokens to the tokenizer (and resize embeddings and lm_head)
NUM_BINS = 64
for i in range(NUM_BINS):
    processor.tokenizer.add_tokens(f"[action_{i}]")
model.resize_token_embeddings(len(processor.tokenizer))

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# print trainable parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)
# breakpoint()

# for name, param in model.visual.merger.named_parameters():
#     print(name, param.shape)

# [name for name, param in model.visual.merger.named_parameters()]

image_path = "/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/images"
traj_path = "/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/success_trajs_smooth_target"

samples = []
for traj_id in os.listdir(image_path):
    # load in trajectory txt
    traj = []
    with open(os.path.join(traj_path, f"{traj_id}.txt"), "r") as f:
        for line in f:
            traj.append(eval(line))

    if len(traj) != len(os.listdir(os.path.join(image_path, traj_id))):
        print(f"Trajectory {traj_id} is incomplete, skipping...")
        continue
    
    for i in range(len(traj)-1):
        sample = {}
        sample["traj_id"] = traj_id
        sample["frame_id"] = str(i)
        sample["current_pose"] = [round(j, 4) for j in traj[i]]
        sample["target_coordinate"] = [-83.9409, 85.6762]
        sample["delta"] = [round(traj[i+1][j] - traj[i][j], 4) for j in range(3)]
        samples.append(sample)

# calculate 1% and 99% percentile of each action dimension
all_actions = np.array([samples[i]["delta"] for i in range(len(samples))])
action_percentile_1 = np.percentile(all_actions, 1, axis=0)
action_percentile_99 = np.percentile(all_actions, 99, axis=0)

print(action_percentile_1)
print(action_percentile_99)

# save the percentiles
np.save("/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/delta_percentile_1.npy", action_percentile_1)
np.save("/home/junru/UAV-Flow/OpenVLA_reproduce-main/ppo_data/datasets--Woody928--phase_1_single_initial/snapshots/db28b5383d1a2454ac6a63c74014a487c04bcc07/delta_percentile_99.npy", action_percentile_99)

action_tokenizers = []
for i in range(all_actions.shape[1]):
    action_tokenizers.append(ActionTokenizer(bins=NUM_BINS, tokenizer=processor.tokenizer, 
                                             min_action=action_percentile_1[i], max_action=action_percentile_99[i]))

for sample in samples:
    action = ""
    for j in range(all_actions.shape[1]):
        action += action_tokenizers[j].tokenize([sample["delta"][j]])[0]
    # print(action)
    sample["label"] = [action]

# process the data using multithreading
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(format_data_ppo, samples[i], None) for i in range(len(samples))]
    train_dataset = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing training set..."):
        train_dataset.append(future.result())

# randomly select 1000 samples from the training set as validation set (set random seed to 928)
random.seed(928)
val_dataset = random.sample(train_dataset, 1000)
train_dataset = [sample for sample in train_dataset if sample not in val_dataset]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn_for_trainer,
    tokenizer=processor.tokenizer,
)

# trainer.train(resume_from_checkpoint="/mnt/data/songjunru/uav_flow_ckpt/1step_delta/checkpoint-19000/")
# trainer.train(resume_from_checkpoint="/mnt/data/songjunru/phase_1_single_initial_ckpt/checkpoint-4000/")
trainer.train(resume_from_checkpoint="/mnt/data/songjunru/phase_1_single_initial_64_ckpt/checkpoint-10000/")
