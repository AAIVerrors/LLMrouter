import json
from huggingface_hub import hf_hub_download

fname = hf_hub_download(
        repo_id="lmarena-ai/p2l-135m-grk-01112025", filename="model_list.json", repo_type="model"
    )

with open(fname) as fin:
    model_list = json.load(fin)

print(model_list)