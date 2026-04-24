#!/bin/bash

set -e

SPARSENE_END2END_DIR=$(pwd)
SPARSENE_ROOT=$(cd "$SPARSENE_END2END_DIR/.." && pwd)
SPARSENE_AD_ROOT=$(cd "$SPARSENE_ROOT/.." && pwd)

python3 -m pip install -U virtualenv
python3 -m virtualenv $SPARSENE_AD_ROOT/venv_dgl
source $SPARSENE_AD_ROOT/venv_dgl/bin/activate

python -m pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

python -m pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu124/repo.html

$SPARSENE_AD_ROOT/venv_dgl/bin/python - <<'PY'
import torch
import dgl
import dgl.nn.pytorch as dglnn

print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)

dev = torch.device('cuda:0')
g = dgl.graph((torch.tensor([0], device=dev), torch.tensor([0], device=dev)), num_nodes=1, device=dev)
x = torch.ones((1, 1), device=dev)
conv = dglnn.GraphConv(1, 1, norm='none', weight=False, bias=False, allow_zero_in_degree=True).to(dev)
_ = conv(g, x)
print('DGL CUDA check: OK')
PY