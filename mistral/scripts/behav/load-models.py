from nnsight import LanguageModel
from transformers import BitsAndBytesConfig
import torch
import os
models = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-v0.3": "mistralai/Mistral-7B-v0.3",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "gpt-2-xl": "openai-community/gpt2-xl",
    "qwen-2-0.5b": "Qwen/Qwen2-0.5B",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "qwen-2-1.5b": "Qwen/Qwen2-0.5B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B"
}
os.environ['HF_HOME'] = "/mnt/align4_drive/data/huggingface"
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
for model_name in models.keys():
    if '7b' in model_name or '8b' in model_name or '70b' in model_name or 'v0.3' in model_name:
        print('Running quantized model')
        model = LanguageModel(models[model_name], 
            device_map='auto', 
            quantization_config=nf4_config
        )
    else:
        model = LanguageModel(models[model_name], 
            device_map='auto'
        )
    print('###### MODEL NAME ', model_name)
    print(model.config)
    del model 