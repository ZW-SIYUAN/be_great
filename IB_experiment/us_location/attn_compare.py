"""
对比两种FD在最强头上的注意力分布差异
state_code -> bird  (查表型FD)
lat        -> lat_zone (阈值型FD)
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, GPT2LMHeadModel
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

base    = 'gpt2-medium'
adapter = 'c:/reposity_clone/be_great/IB_experiment/us_location/us_great_model/lora_adapter'

tokenizer = AutoTokenizer.from_pretrained(base)
model     = GPT2LMHeadModel.from_pretrained(base, output_attentions=True)
model     = PeftModel.from_pretrained(model, adapter)
model.eval()

# ── 测试多个样本, 验证FD头是否稳定 ──────────────────────────────────────
EXAMPLES = [
    ('state_code is MA, lat is 42.47, lon is -72.65, bird is Black-capped Chickadee, lat_zone is middle',
     'MA', ' MA', 42.47, 'middle', ' middle'),
    ('state_code is AK, lat is 65.10, lon is -152.30, bird is Willow Ptarmigan, lat_zone is high',
     'AK', ' AK', 65.10, 'high', ' high'),
    ('state_code is HI, lat is 20.50, lon is -156.40, bird is Hawaiian Nene, lat_zone is low',
     'HI', ' HI', 20.50, 'low', ' low'),
    ('state_code is CA, lat is 36.60, lon is -119.80, bird is California Quail, lat_zone is middle',
     'CA', ' CA', 36.60, 'middle', ' middle'),
]

# ── Best heads identified: L16H2 for bird-FD, L21H1 for zone-FD ──────────
BIRD_L, BIRD_H = 16, 2
ZONE_L, ZONE_H = 21, 1

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
fig.suptitle('FD Attention Patterns: state_code→bird (lookup) vs lat→lat_zone (threshold)\n'
             'Blue vertical line = FD source token | Red horizontal line = FD target token',
             fontsize=12, y=1.01)

for row, (text, sc_str, sc_tok, lat_val, zone_str, zone_tok) in enumerate(EXAMPLES):
    tokens = tokenizer(text, return_tensors='pt')
    ids    = tokens['input_ids'][0]
    toks   = [tokenizer.decode([t]) for t in ids]

    with torch.no_grad():
        out = model(**tokens)
    attns = out.attentions  # 24 × (1, 16, seq, seq)

    # Identify positions
    sc_pos   = [i for i, t in enumerate(toks) if t == sc_tok]
    lat_pos  = [i for i, t in enumerate(toks) if t.strip().lstrip('-').replace('.','').isdigit()
                and i > 4 and i < 12][:1]  # first number after lat
    bird_pos = [i for i, t in enumerate(toks) if t not in (',', ' is', ' lat', ' lon', ' bird',
                ' lat_zone', ' middle', ' high', ' low', sc_tok) and len(t.strip()) > 1
                and i >= 20 and i < 30]
    zone_pos = [i for i, t in enumerate(toks) if t == zone_tok]

    seq_len = len(toks)

    # ── Col 0: Bird-FD best head attention matrix ──────────────────────────
    ax = axes[row, 0]
    ha = attns[BIRD_L][0, BIRD_H].numpy()
    im = ax.imshow(ha, cmap='Oranges', aspect='auto', vmin=0, vmax=ha.max())
    ax.set_title(f'[{sc_str}] Bird-FD  L{BIRD_L}H{BIRD_H}', fontsize=9)
    ax.set_xticks(range(seq_len)); ax.set_xticklabels(toks, rotation=90, fontsize=5)
    ax.set_yticks(range(seq_len)); ax.set_yticklabels(toks, fontsize=5)
    for i in sc_pos:
        ax.axvline(i, color='royalblue', linewidth=2.5, alpha=0.8)
    for i in bird_pos:
        ax.axhline(i, color='crimson',   linewidth=2.5, alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Col 1: Zone-FD best head attention matrix ──────────────────────────
    ax = axes[row, 1]
    ha = attns[ZONE_L][0, ZONE_H].numpy()
    im = ax.imshow(ha, cmap='Greens', aspect='auto', vmin=0, vmax=ha.max())
    ax.set_title(f'[{sc_str}] Zone-FD  L{ZONE_L}H{ZONE_H}', fontsize=9)
    ax.set_xticks(range(seq_len)); ax.set_xticklabels(toks, rotation=90, fontsize=5)
    ax.set_yticks(range(seq_len)); ax.set_yticklabels(toks, fontsize=5)
    for i in lat_pos:
        ax.axvline(i, color='royalblue', linewidth=2.5, alpha=0.8)
    for i in zone_pos:
        ax.axhline(i, color='crimson',   linewidth=2.5, alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Col 2: Bird target row - attention distribution ────────────────────
    ax = axes[row, 2]
    ha_bird = attns[BIRD_L][0, BIRD_H].numpy()
    for bp in bird_pos[:1]:
        ax.bar(range(seq_len), ha_bird[bp], color='salmon', alpha=0.8)
        ax.set_title(f'[{sc_str}] Row={repr(toks[bp])} attn dist (Bird head)', fontsize=8)
    for i in sc_pos:
        ax.axvline(i, color='royalblue', linewidth=2, linestyle='--', alpha=0.9,
                   label='state_code value')
    ax.set_xticks(range(seq_len)); ax.set_xticklabels(toks, rotation=90, fontsize=5)
    ax.set_ylabel('Attn weight'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)

    # ── Col 3: Zone target row - attention distribution ────────────────────
    ax = axes[row, 3]
    ha_zone = attns[ZONE_L][0, ZONE_H].numpy()
    for zp in zone_pos:
        ax.bar(range(seq_len), ha_zone[zp], color='lightgreen', alpha=0.8)
        ax.set_title(f'[{sc_str}] Row={repr(toks[zp])} attn dist (Zone head)', fontsize=8)
    for i in lat_pos:
        ax.axvline(i, color='royalblue', linewidth=2, linestyle='--', alpha=0.9,
                   label='lat value')
    ax.set_xticks(range(seq_len)); ax.set_xticklabels(toks, rotation=90, fontsize=5)
    ax.set_ylabel('Attn weight'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)

plt.tight_layout()
out_path = 'c:/reposity_clone/be_great/IB_experiment/us_location/results/fd_attention_compare.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f'Saved: {out_path}')

# ── Print summary stats ──────────────────────────────────────────────────
print('\n=== FD Score Summary (across 4 examples) ===')
print(f'{"Example":<15} {"Bird-FD score":>14} {"Zone-FD score":>14}')
for text, sc_str, sc_tok, lat_val, zone_str, zone_tok in EXAMPLES:
    tokens = tokenizer(text, return_tensors='pt')
    ids    = tokens['input_ids'][0]
    toks   = [tokenizer.decode([t]) for t in ids]
    with torch.no_grad():
        out = model(**tokens)
    attns = out.attentions
    sc_pos  = [i for i, t in enumerate(toks) if t == sc_tok]
    lat_pos = [i for i, t in enumerate(toks) if t.strip().lstrip('-').replace('.','').isdigit()
               and i > 4 and i < 12][:1]
    bird_pos= [i for i, t in enumerate(toks) if t not in (',', ' is', ' lat', ' lon', ' bird',
                ' lat_zone', ' middle', ' high', ' low', sc_tok) and len(t.strip()) > 1
                and i >= 20 and i < 30]
    zone_pos= [i for i, t in enumerate(toks) if t == zone_tok]

    ha_bird = attns[BIRD_L][0, BIRD_H].numpy()
    ha_zone = attns[ZONE_L][0, ZONE_H].numpy()

    bird_score = float(ha_bird[np.ix_(bird_pos, sc_pos)].mean()) if bird_pos and sc_pos else 0
    zone_score = float(ha_zone[np.ix_(zone_pos, lat_pos)].mean()) if zone_pos and lat_pos else 0
    print(f'{sc_str:<15} {bird_score:>14.4f} {zone_score:>14.4f}')
