import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
M="GSAI-ML/LLaDA-8B-Instruct"; MASK=126336
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(M,trust_remote_code=True,torch_dtype=torch.bfloat16).cuda().eval()
def ids(s): return tok.encode(s,add_special_tokens=False)
@torch.no_grad()
def fd(seq,pos):
    out=model(torch.tensor([seq]).cuda()); lg=out.logits if hasattr(out,"logits") else out[0]
    return torch.softmax(lg[0,pos].float(),-1).cpu().numpy()

def find_tokens(sent_ids, word):
    """返回 word(单token,带前导空格)在句子里的所有位置"""
    wid=ids(word)
    if len(wid)!=1: return []
    return [i for i,t in enumerate(sent_ids) if t==wid[0]]

def test(sentence, target_words, n=400):
    sent=ids(sentence)
    pos=[]
    for w in target_words: pos+=find_tokens(sent,w)
    pos=sorted(set(pos))
    if len(pos)<2: print(f"  [skip] '{sentence[:30]}' 找不到足够目标词"); return
    base=list(sent)
    for p in pos: base[p]=MASK
    # 每个位置单独 conf（其余仍 mask）
    confs={p:fd(base,p).max() for p in pos}
    # 只保留本身高 conf 的位置（排除内容不确定）
    keep=[p for p in pos if confs[p]>0.7]
    if len(keep)<2:
        print(f"  [skip] '{sentence[:30]}' 高conf位置不足({[round(confs[p],2) for p in pos]})"); return
    base=list(sent)
    for p in keep: base[p]=MASK
    margs={p:fd(base,p) for p in keep}
    confs={p:margs[p].max() for p in keep}
    min_conf=min(confs.values())
    # 顺序 greedy 一致解
    s=list(base); seq_lp=0.0
    for p in keep:
        d=fd(s,p); t=int(np.argmax(d)); seq_lp+=np.log(d[t]+1e-12); s[p]=t
    # 并行 argmax 解，在顺序联合下评分
    par={p:int(np.argmax(margs[p])) for p in keep}
    s=list(base); par_lp=0.0
    for p in keep:
        d=fd(s,p); par_lp+=np.log(d[par[p]]+1e-12); s[p]=par[p]
    gap=seq_lp-par_lp
    # 蒙特卡洛坏组合率
    rng=np.random.default_rng(0); bad=0
    for _ in range(n):
        draw={p:int(rng.choice(len(margs[p]),p=margs[p])) for p in keep}
        s=list(base); lp=0.0
        for p in keep:
            d=fd(s,p); lp+=np.log(d[draw[p]]+1e-12); s[p]=draw[p]
        if lp<seq_lp-np.log(10): bad+=1
    err=bad/n
    flag="  <<< 高conf+高失真!!!" if (min_conf>0.85 and (gap>1 or err>0.2)) else ""
    print(f"  k={len(keep)} min_conf={min_conf:.3f}  gap={gap:.2f}  坏组合率={err*100:.1f}%  '{sentence[:40]}...'{flag}")

print("=== 多槽(只测高conf位置) ===")
test("the actress said she would bring her own car and her own bag", [" her"," she"])
test("the king knew that he would lose his crown and his throne soon", [" he"," his"])
test("he was tired and she was tired and they were also tired", [" tired"])
test("the two big red books are on the three small wooden shelves", [" the"," are"])
test("neither the manager nor the workers were willing to sign it", [" the"])
test("she put her own coat on her own chair near her own desk", [" her"," own"])
test("if it rains then it floods and if it floods then it stops", [" it"," then"])
print("\n看有没有 <<< 标记:有=高维下confidence兜不住,方向复活;无=收")