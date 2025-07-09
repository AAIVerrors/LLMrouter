from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, login
import json 
from quality_model import P2LPredictor

# Initialize once
predictor = P2LPredictor()

# Get coefficients for a prompt
coefs = predictor.get_coefficients("Explain quantum computing")
# Returns: {'model1': 0.85, 'model2': 0.23, ...}

print(coefs)

