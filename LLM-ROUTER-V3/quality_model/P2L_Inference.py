import os
import torch
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from P2L import P2LModel, RKHead
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer


def setup_hf_authentication(token=None):
    """
    Set up Hugging Face authentication.
    
    Args:
        token (str): Hugging Face access token. If None, will try to use environment variable.
    """
    if token:
        login(token)
    elif os.getenv('HUGGING_FACE_HUB_TOKEN'):
        login(os.getenv('HUGGING_FACE_HUB_TOKEN'))
    else:
        print("Warning: No Hugging Face token provided. Some datasets may not be accessible.")
        print("Set HUGGING_FACE_HUB_TOKEN environment variable or pass token to setup_hf_authentication()")

def load_chatbot_arena_data(split="train", max_samples=None):
    """
    Load Chatbot Arena conversations dataset.
    
    Args:
        split (str): Dataset split to load ("train", "test", "validation")
        max_samples (int): Maximum number of samples to load (None for all)
    
    Returns:
        dataset: HuggingFace dataset object
    """
    print(f"Loading Chatbot Arena dataset ({split} split)...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} conversations")
    return dataset

def extract_prompts_from_conversations(dataset, max_prompts=None):
    """
    Extract individual prompts from conversations.
    
    Args:
        dataset: Chatbot Arena dataset
        max_prompts (int): Maximum number of prompts to extract
    
    Returns:
        list: List of prompt strings
    """
    prompts = []
    
    for conversation in tqdm(dataset, desc="Extracting prompts"):
        # Extract prompts from conversation_a (first message is usually the human prompt)
        if 'conversation_a' in conversation:
            conv_a = conversation['conversation_a']
            if isinstance(conv_a, list) and len(conv_a) > 0:
                # Get the first message (human prompt)
                first_message = conv_a[0]
                if isinstance(first_message, dict) and 'content' in first_message:
                    prompts.append(first_message['content'])
        
        # Also extract from conversation_b if it has a different prompt
        if 'conversation_b' in conversation:
            conv_b = conversation['conversation_b']
            if isinstance(conv_b, list) and len(conv_b) > 0:
                first_message = conv_b[0]
                if isinstance(first_message, dict) and 'content' in first_message:
                    # Only add if it's different from conversation_a
                    if len(prompts) == 0 or prompts[-1] != first_message['content']:
                        prompts.append(first_message['content'])
    
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    print(f"Extracted {len(prompts)} prompts")
    if len(prompts) > 0:
        print(f"First prompt example: {prompts[0][:100]}...")
    
    return prompts

def batch_predict_model_performance(prompts, model, tokenizer, model_list, batch_size=8):
    """
    Perform batch inference on multiple prompts.
    
    Args:
        prompts (list): List of prompt strings
        model: The loaded P2L model
        tokenizer: The tokenizer
        model_list (list): List of model names
        batch_size (int): Batch size for inference
    
    Returns:
        list: List of prediction results for each prompt
    """
    model.eval()
    device = next(model.parameters()).device
    results = []
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True  # Ensure special tokens are added
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Ensure CLS token is present at the beginning of each sequence
        batch_size_actual = inputs['input_ids'].shape[0]
        cls_token_id = tokenizer.cls_token_id
        
        # Check if CLS token is already present, if not add it
        for j in range(batch_size_actual):
            if inputs['input_ids'][j][0] != cls_token_id:
                # Add CLS token at the beginning
                cls_tensor = torch.tensor([[cls_token_id]], device=device)
                # Ensure both tensors have the same number of dimensions
                current_ids = inputs['input_ids'][j].unsqueeze(0)  # Add batch dimension
                current_mask = inputs['attention_mask'][j].unsqueeze(0)  # Add batch dimension
                
                # Concatenate and remove the extra dimension
                inputs['input_ids'][j] = torch.cat([cls_tensor, current_ids[:, :-1]], dim=1).squeeze(0)
                inputs['attention_mask'][j] = torch.cat([torch.tensor([[1]], device=device), current_mask[:, :-1]], dim=1).squeeze(0)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process each sample in the batch - convert BFloat16 to float32 before numpy
        batch_coefs = outputs.coefs.float().cpu().numpy()  # Shape: (batch_size, num_models)
        batch_eta = outputs.eta.float().cpu().numpy() if outputs.eta is not None else None
        
        for j, (prompt, coefs) in enumerate(zip(batch_prompts, batch_coefs)):
            # Create model rankings for this prompt
            model_scores = list(zip(model_list, coefs))
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            result = {
                'prompt': prompt,
                'coefficients': dict(zip(model_list, coefs)),
                'eta': batch_eta[j] if batch_eta is not None else None,
                'rankings': model_scores,
                'top_model': model_scores[0] if model_scores else None
            }
            results.append(result)
    
    return results

def save_raw_results(results, output_dir="./p2l_results"):
    """
    Save raw prediction results in JSON format.
    
    Args:
        results (list): Prediction results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            return super(NumpyEncoder, self).default(obj)
    
    # Save raw results as JSON using custom encoder
    with open(f"{output_dir}/raw_predictions_top5.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Raw results saved to {output_dir}/raw_predictions.json")

# Main execution function
def run_p2l_inference_on_arena_data(model, tokenizer, model_list, max_samples=1000, max_prompts=500, batch_size=8):
    """
    Run P2L inference on Chatbot Arena dataset.
    
    Args:
        model: The loaded P2L model
        tokenizer: The tokenizer
        model_list (list): List of model names
        max_samples (int): Maximum number of conversations to load
        max_prompts (int): Maximum number of prompts to process
        batch_size (int): Batch size for inference
    """
    print("Starting P2L inference on Chatbot Arena dataset...")
    print(f"Model list contains {len(model_list)} models")
    print(f"First 5 models: {model_list[:5]}")
    
    # Load dataset
    dataset = load_chatbot_arena_data(split="train", max_samples=max_samples)
    
    # Extract prompts
    prompts = extract_prompts_from_conversations(dataset, max_prompts=max_prompts)
    
    # Run predictions
    results = batch_predict_model_performance(prompts, model, tokenizer, model_list, batch_size=batch_size)
    
    # Save raw results
    save_raw_results(results)
    
    # Print simple summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Processed {len(results)} prompts")
    print(f"Raw results saved to ./p2l_results/raw_predictions.json")
    
    return results

if __name__ == "__main__":
    # Set up Hugging Face authentication
    setup_hf_authentication("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")

    # Load model and tokenizer
    fname = hf_hub_download(
            repo_id="lmarena-ai/p2l-7b-grk-02222025", filename="model_list.json", repo_type="model"
        )

    with open(fname) as fin:
        model_list = json.load(fin)
       


    print(f"Model list: {model_list}")


    tokenizer = AutoTokenizer.from_pretrained("lmarena-ai/p2l-7b-grk-02222025")
    model = P2LModel.from_pretrained(
        "lmarena-ai/p2l-7b-grk-02222025",
        CLS_id=tokenizer.cls_token_id,
        num_models=len(model_list),
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Example usage
    # Run inference on a subset of the dataset
    results = run_p2l_inference_on_arena_data(
        model=model,           # Pass model as argument
        tokenizer=tokenizer,   # Pass tokenizer as argument
        model_list=model_list, # Pass model_list as argument
        max_samples=200,       # Load 100 conversations
        max_prompts=100,       # Process 200 prompts
        batch_size=16           # Use batch size of 4
    )