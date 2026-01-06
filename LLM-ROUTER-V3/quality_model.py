import os
import torch
import json
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer
from P2L import P2LModel


def setup_hf_authentication(token=None):
    """
    Set up Hugging Face authentication.
    
    Args:
        token (str): Hugging Face access token. If None, will try to use environment variable.
    """
    if token:
        login(token)
    elif os.getenv('HF_TOKEN'):
        login(os.getenv('HF_TOKEN'))
    else:
        print("Warning: No Hugging Face token provided. Some datasets may not be accessible.")
        print("Set HF_TOKEN environment variable or pass token to setup_hf_authentication()")


class P2LPredictor:
    """
    P2L model wrapper for single prompt inference
    """
    
    def __init__(self, repo_id="lmarena-ai/p2l-0.5b-grk-01112025", hf_token=None):
        """
        Initialize the P2L predictor
        
        Args:
            repo_id (str): Hugging Face repository ID
            hf_token (str): Hugging Face token for authentication
        """
        # Set up authentication
        setup_hf_authentication(hf_token)
        
        # Load model list
        fname = hf_hub_download(
            repo_id=repo_id, 
            filename="model_list.json", 
            repo_type="model"
        )
        
        with open(fname) as fin:
            self.model_list = json.load(fin)
        
        # print(f"Loaded {len(self.model_list)} models: {self.model_list}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = P2LModel.from_pretrained(
            repo_id,
            CLS_id=self.tokenizer.cls_token_id,
            num_models=len(self.model_list),
            torch_dtype=torch.bfloat16,
        )
        
        # Set device and move model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, prompt):
        """
        Predict model performance coefficients for a single prompt
        
        Args:
            prompt (str): Input prompt text
            
        Returns:
            dict: Dictionary containing coefficients and model rankings
        """
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Ensure CLS token is present at the beginning
        cls_token_id = self.tokenizer.cls_token_id
        if inputs['input_ids'][0][0] != cls_token_id:
            # Add CLS token at the beginning
            cls_tensor = torch.tensor([[cls_token_id]], device=self.device)
            # Concatenate and adjust
            inputs['input_ids'] = torch.cat([cls_tensor, inputs['input_ids'][:, :-1]], dim=1)
            inputs['attention_mask'] = torch.cat([torch.tensor([[1]], device=self.device), 
                                                inputs['attention_mask'][:, :-1]], dim=1)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert outputs to numpy (handle BFloat16)
        coefs = outputs.coefs.float().cpu().numpy().flatten()  # Shape: (num_models,)
        eta = outputs.eta.float().cpu().numpy().item() if outputs.eta is not None else None
        
        # Create coefficient dictionary
        coefficients = dict(zip(self.model_list, coefs))
        
        # Create model rankings
        model_scores = list(zip(self.model_list, coefs))
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'prompt': prompt,
            'coefficients': coefficients,
            'eta': eta,
            'rankings': model_scores,
            'top_model': model_scores[0] if model_scores else None,
            'top_3_models': model_scores[:3] if len(model_scores) >= 3 else model_scores
        }
        
        return result
    
    def get_coefficients(self, prompt):
        """
        Simple method to get just the coefficients
        
        Args:
            prompt (str): Input prompt text
            
        Returns:
            dict: Dictionary mapping model names to coefficients
        """
        result = self.predict(prompt)
        return result['coefficients']
    
    def get_top_model(self, prompt):
        """
        Get the top recommended model for a prompt
        
        Args:
            prompt (str): Input prompt text
            
        Returns:
            tuple: (model_name, coefficient)
        """
        result = self.predict(prompt)
        return result['top_model']


def predict_single_prompt(prompt, repo_id="lmarena-ai/p2l-135m-bt-01132025", hf_token=None):
    """
    Convenience function for single prompt prediction
    
    Args:
        prompt (str): Input prompt text
        repo_id (str): Hugging Face repository ID
        hf_token (str): Hugging Face token
        
    Returns:
        dict: Prediction results
    """
    predictor = P2LPredictor(repo_id=repo_id, hf_token=hf_token)
    return predictor.predict(prompt)

