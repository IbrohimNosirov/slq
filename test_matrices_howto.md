# Retrieving Large Matrices for Eigensolver Benchmarking

## 1. Transformer Q, K, V Matrices

Extract projection weight matrices from pretrained models on [Hugging Face](https://huggingface.co/models).

```bash
pip install transformers torch
```

```python
from transformers import AutoModel

# GPT-2: W_Q, W_K, W_V are concatenated in c_attn (768 x 2304)
model = AutoModel.from_pretrained("gpt2")
qkv = model.h[0].attn.c_attn.weight  # layer 0
W_Q, W_K, W_V = qkv.chunk(3, dim=-1)  # each 768 x 768

# BERT: separate Q, K, V per layer
model = AutoModel.from_pretrained("bert-base-uncased")
W_Q = model.encoder.layer[0].attention.self.query.weight  # 768 x 768
W_K = model.encoder.layer[0].attention.self.key.weight
W_V = model.encoder.layer[0].attention.self.value.weight

# For larger matrices, use bigger models:
#   gpt2-medium (1024), gpt2-large (1280), gpt2-xl (1600)
#   meta-llama/Llama-2-7b-hf (4096), meta-llama/Llama-2-70b-hf (8192)
```

To get the **attention matrix** (n×n, where n = sequence length) rather than the weight matrices:

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", output_attentions=True)
inputs = tokenizer("Your input text here " * 100, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
attn_matrix = outputs.attentions[0][0, 0]  # layer 0, head 0 (seq_len x seq_len)
```

---

## 2. Neural Network Hessian Matrix

Use [PyHessian](https://github.com/amirgholami/PyHessian) to compute Hessian information from any trained model.

```bash
git clone https://github.com/amirgholami/PyHessian.git
cd PyHessian
pip install -e .
```

```python
from pyhessian import hessian
import torch
import torchvision.models as models
from torchvision import datasets, transforms

# Load a trained model (e.g., ResNet-18 on CIFAR-10)
model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load("your_checkpoint.pth"))
model.eval()

# Prepare a batch of data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False)
inputs, targets = next(iter(loader))

# Compute Hessian properties (matrix-free, no explicit formation needed)
criterion = torch.nn.CrossEntropyLoss()
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)

top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=10)
trace = hessian_comp.trace()
density_eigen, density_weight = hessian_comp.density()
```

> **Note:** Full Hessian matrices are too large to store explicitly for most networks
> (e.g., ResNet-18 has ~11M parameters → Hessian is 11M × 11M). PyHessian uses
> Hessian-vector products to compute eigenvalues, traces, and spectral densities
> without forming the full matrix. If you need a materialised Hessian, use a small
> network (e.g., a 2-layer MLP on MNIST with ~7,840 parameters → 7,840 × 7,840 Hessian).

---

## 3. Social Network Community Matrices

### Option A: SNAP (edge lists)

Download from [https://snap.stanford.edu/data/](https://snap.stanford.edu/data/)

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| com-DBLP | 317K | 1.0M | DBLP collaboration, ground-truth communities |
| com-Amazon | 335K | 926K | Amazon co-purchase, ground-truth communities |
| com-YouTube | 1.1M | 3.0M | YouTube social network with communities |
| com-LiveJournal | 4.0M | 34.7M | LiveJournal social network with communities |
| ego-Facebook | 4K | 88K | Facebook ego networks |

```bash
# Example: download com-DBLP
wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
gunzip com-dblp.ungraph.txt.gz
# Ground-truth communities:
wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz
```

Convert edge list to sparse adjacency matrix:

```python
import numpy as np
from scipy.sparse import coo_matrix

edges = np.loadtxt("com-dblp.ungraph.txt", comments="#", dtype=int)
n = edges.max() + 1
A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
A = A + A.T  # symmetrise
A = (A > 0).astype(float)  # binary adjacency
```

### Option B: SuiteSparse Matrix Collection (ready-made sparse matrices)

Download directly from [https://sparse.tamu.edu/SNAP](https://sparse.tamu.edu/SNAP) in Matrix Market or MATLAB `.mat` format.

```python
from scipy.io import mmread
# After downloading a .mtx file:
A = mmread("com-DBLP.mtx")  # returns scipy sparse matrix
```

Or use `ssgetpy`:

```bash
pip install ssgetpy
```

```python
import ssgetpy
results = ssgetpy.search(group="SNAP", rowbounds=(1000, 5000000))
results[0].download(destpath="./matrices", format="MM")
```

### Option C: MIT GraphChallenge (S3-hosted, matrix formats)

Browse: [https://graphchallenge.mit.edu/data-sets](https://graphchallenge.mit.edu/data-sets)

```bash
# Example: download Facebook ego network adjacency matrix
wget https://graphchallenge.s3.amazonaws.com/snap/ego-Facebook/ego-Facebook_adj.tsv
# Or in Matrix Market format:
wget https://graphchallenge.s3.amazonaws.com/snap/ego-Facebook/ego-Facebook_adj.mmio
```
