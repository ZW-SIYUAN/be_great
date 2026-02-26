import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, GPT2LMHeadModel
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

base = 'gpt2-medium'
adapter = 'c:/reposity_clone/be_great/IB_experiment/us_location/us_great_model/lora_adapter'

tokenizer = AutoTokenizer.from_pretrained(base)
model = GPT2LMHeadModel.from_pretrained(base, output_attentions=True)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

# Two example rows with different FD values
examples = [
    # FD1: state_code=MA -> bird=Black-capped Chickadee
    # FD2: lat=42.47     -> lat_zone=middle
    'state_code is MA, lat is 42.47, lon is -72.65, bird is Black-capped Chickadee, lat_zone is middle',
    # FD1: state_code=AK -> bird=Willow Ptarmigan
    # FD2: lat=65.10     -> lat_zone=high
    'state_code is AK, lat is 65.10, lon is -152.30, bird is Willow Ptarmigan, lat_zone is high',
    # FD1: state_code=HI -> bird=Hawaiian Nene
    # FD2: lat=20.50     -> lat_zone=low
    'state_code is HI, lat is 20.50, lon is -156.40, bird is Hawaiian Nene, lat_zone is low',
]

# ── 1. Tokenize and show token list ─────────────────────────────────────────
text = examples[0]
tokens = tokenizer(text, return_tensors='pt')
ids = tokens['input_ids'][0]
token_strs = [tokenizer.decode([t]) for t in ids]
print('Token list:')
for i, t in enumerate(token_strs):
    print(f'  {i:2d}: {repr(t)}')

# Mark key token index ranges (approximate)
# "state_code is MA" -> find "MA" token
# "lat is 42.47"     -> find "42" "." "47" tokens
# "bird is Black"    -> find bird name tokens
# "lat_zone is middle" -> find "middle" token

def find_span(token_strs, keyword):
    """Find all token indices containing keyword substring"""
    return [i for i, t in enumerate(token_strs) if keyword.lower() in t.lower()]

# ── 2. Run model and extract attention ─────────────────────────────────────
with torch.no_grad():
    out = model(**tokens)

# attentions: tuple of (1, num_heads, seq, seq) per layer
# GPT2-medium: 24 layers, 16 heads
attentions = out.attentions  # 24 tensors
seq_len = len(ids)
num_layers = len(attentions)
num_heads  = attentions[0].shape[1]
print(f'\nModel: {num_layers} layers, {num_heads} heads, seq_len={seq_len}')

# ── 3. Compute per-head attention from each "target col value" to "source col value" ─
# For FD lat->lat_zone: source=lat value tokens, target=lat_zone value tokens
# For FD state_code->bird: source=state_code value, target=bird value

# Identify token positions for key column values
sc_val  = find_span(token_strs, 'MA')        # state_code value
lat_val = [i for i, t in enumerate(token_strs) if any(c.isdigit() for c in t) and i > 3 and i < 10]
bird_val= find_span(token_strs, 'Chick') + find_span(token_strs, 'capped') + find_span(token_strs, 'Black')
zone_val= find_span(token_strs, 'middle')

print(f'\nstate_code val tokens : {sc_val}   -> {[token_strs[i] for i in sc_val]}')
print(f'lat val tokens        : {lat_val}  -> {[token_strs[i] for i in lat_val]}')
print(f'bird val tokens       : {bird_val} -> {[token_strs[i] for i in bird_val]}')
print(f'lat_zone val tokens   : {zone_val} -> {[token_strs[i] for i in zone_val]}')

# ── 4. Average attention across layers & heads, show the full matrix ──────
attn_avg = torch.stack([a[0] for a in attentions]).mean(dim=0)  # (heads, seq, seq)
attn_avg_all = attn_avg.mean(dim=0).numpy()  # (seq, seq)

# ── 5. Plot: full avg attention matrix + head-specific patterns ────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('GPT-2 Medium Attention Patterns on GReaT Tabular Text\n'
             f'"{text}"', fontsize=10, wrap=True)

# Plot 1: full average attention matrix
ax = axes[0, 0]
im = ax.imshow(attn_avg_all, cmap='Blues', aspect='auto', vmin=0)
ax.set_title('Avg across all layers & heads', fontsize=9)
ax.set_xticks(range(seq_len))
ax.set_xticklabels(token_strs, rotation=90, fontsize=5)
ax.set_yticks(range(seq_len))
ax.set_yticklabels(token_strs, fontsize=5)
plt.colorbar(im, ax=ax)

# Mark FD target rows
for idx in zone_val:
    ax.axhline(idx - 0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(idx + 0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
for idx in bird_val:
    ax.axhline(idx - 0.5, color='orange', linewidth=1, linestyle='--', alpha=0.7)
    ax.axhline(idx + 0.5, color='orange', linewidth=1, linestyle='--', alpha=0.7)

# Find heads with strongest FD signals
# FD1: state_code->bird: attention from bird_val rows to sc_val cols
# FD2: lat->lat_zone:    attention from zone_val rows to lat_val cols
def fd_score(head_attn, src_cols, tgt_rows):
    """Average attention weight from target rows to source columns"""
    if not src_cols or not tgt_rows:
        return 0.0
    return float(head_attn[np.ix_(tgt_rows, src_cols)].mean())

fd_bird_scores = []  # per (layer, head)
fd_zone_scores = []
for l in range(num_layers):
    for h in range(num_heads):
        ha = attentions[l][0, h].numpy()
        fd_bird_scores.append((fd_score(ha, sc_val, bird_val), l, h))
        fd_zone_scores.append((fd_score(ha, lat_val, zone_val), l, h))

fd_bird_scores.sort(reverse=True)
fd_zone_scores.sort(reverse=True)

print('\nTop-5 heads for state_code -> bird FD:')
for score, l, h in fd_bird_scores[:5]:
    print(f'  Layer {l:2d} Head {h:2d}: score={score:.4f}')

print('\nTop-5 heads for lat -> lat_zone FD:')
for score, l, h in fd_zone_scores[:5]:
    print(f'  Layer {l:2d} Head {h:2d}: score={score:.4f}')

# Plot top-3 bird-FD heads
for k, (score, l, h) in enumerate(fd_bird_scores[:3]):
    ax = axes[1, k]
    ha = attentions[l][0, h].numpy()
    im = ax.imshow(ha, cmap='Oranges', aspect='auto', vmin=0)
    ax.set_title(f'Bird-FD head L{l}H{h}\n(score={score:.4f})', fontsize=9)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_strs, rotation=90, fontsize=5)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(token_strs, fontsize=5)
    plt.colorbar(im, ax=ax)
    # highlight FD source cols (state_code) and target rows (bird)
    for idx in sc_val:
        ax.axvline(idx, color='blue', linewidth=2, alpha=0.5)
    for idx in bird_val:
        ax.axhline(idx, color='red', linewidth=2, alpha=0.5)

# Plot top-3 zone-FD heads
for k, (score, l, h) in enumerate(fd_zone_scores[:3]):
    ax = axes[2, k]
    ha = attentions[l][0, h].numpy()
    im = ax.imshow(ha, cmap='Greens', aspect='auto', vmin=0)
    ax.set_title(f'Zone-FD head L{l}H{h}\n(score={score:.4f})', fontsize=9)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_strs, rotation=90, fontsize=5)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(token_strs, fontsize=5)
    plt.colorbar(im, ax=ax)
    for idx in lat_val:
        ax.axvline(idx, color='blue', linewidth=2, alpha=0.5)
    for idx in zone_val:
        ax.axhline(idx, color='red', linewidth=2, alpha=0.5)

# Bar chart: per-layer average FD score
ax = axes[0, 1]
layer_bird = []
layer_zone = []
for l in range(num_layers):
    b = np.mean([fd_score(attentions[l][0, h].numpy(), sc_val, bird_val) for h in range(num_heads)])
    z = np.mean([fd_score(attentions[l][0, h].numpy(), lat_val, zone_val) for h in range(num_heads)])
    layer_bird.append(b)
    layer_zone.append(z)
x = range(num_layers)
ax.bar(x, layer_bird, alpha=0.6, label='state→bird', color='orange')
ax.bar(x, layer_zone, alpha=0.6, label='lat→zone',   color='green', bottom=layer_bird)
ax.set_xlabel('Layer')
ax.set_ylabel('Avg FD attention score')
ax.set_title('FD Attention Score by Layer', fontsize=9)
ax.legend(fontsize=7)

# Per-head heatmap for best layer
best_l = fd_bird_scores[0][1]
ax = axes[0, 2]
head_bird = [fd_score(attentions[best_l][0, h].numpy(), sc_val, bird_val) for h in range(num_heads)]
head_zone = [fd_score(attentions[best_l][0, h].numpy(), lat_val, zone_val) for h in range(num_heads)]
ax.bar(range(num_heads), head_bird, alpha=0.7, label='state→bird', color='orange')
ax.set_xlabel('Head')
ax.set_ylabel('FD score')
ax.set_title(f'state→bird per head (Layer {best_l})', fontsize=9)

ax = axes[0, 3]
best_l_z = fd_zone_scores[0][1]
head_zone2 = [fd_score(attentions[best_l_z][0, h].numpy(), lat_val, zone_val) for h in range(num_heads)]
ax.bar(range(num_heads), head_zone2, alpha=0.7, label='lat→zone', color='green')
ax.set_xlabel('Head')
ax.set_ylabel('FD score')
ax.set_title(f'lat→zone per head (Layer {best_l_z})', fontsize=9)

# Fill remaining axes
axes[1, 3].axis('off')
axes[2, 3].axis('off')

plt.tight_layout()
out_path = 'c:/reposity_clone/be_great/IB_experiment/us_location/results/fd_attention.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved to {out_path}')
