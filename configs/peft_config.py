from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0,
    r=32,
    bias="none",
    target_modules="all-linear",# ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.0", "mlp.2"],
    modules_to_save=["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
)