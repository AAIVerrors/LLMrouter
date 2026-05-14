from sentence_transformers import SentenceTransformer
import torch, torch.nn.functional as F

m = SentenceTransformer("BAAI/bge-base-en-v1.5")
qs = [
    "Are both The Brothers Creeggan and The Postal Service American bands?",
    "How did Fionn mac Cumhaill win the leadership of the Fianna?",
    "Which magazine was started first, The Delineator or Woman's World?",
    "Who has more scope of profession, Patricia Rozema or John Korty?",
    "What year did the Swiss Bank Corporation merge with UBS?",
]
e = F.normalize(torch.tensor(m.encode(qs)), dim=-1)
print(e @ e.T)