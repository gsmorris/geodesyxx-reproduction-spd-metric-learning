# API Reference

Complete API reference for the Geodesyxx reproduction package.

## Core Modules

### `src.spd_metric`

#### `SPDMetric`

Primary SPD metric tensor implementation using A^T A + εI parameterization.

```python
class SPDMetric(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        rank: int = 16,
        epsilon: float = 1e-6,
        max_condition: float = 1e4,
        device: Optional[torch.device] = None
    )
```

**Parameters:**
- `embedding_dim`: Dimension of embedding space (e.g., 64 for attention heads)  
- `rank`: Rank of A matrix factorization (controls parameter count)
- `epsilon`: Regularization constant for numerical stability
- `max_condition`: Maximum allowed condition number before clipping
- `device`: Device to place tensors on

**Methods:**

##### `get_metric_tensor() -> torch.Tensor`
Compute the SPD metric tensor G = A^T A + εI.

**Returns:** SPD tensor of shape (embedding_dim, embedding_dim)

##### `compute_mahalanobis_distance(x, y, squared=True) -> torch.Tensor`
Compute Mahalanobis distance d_G(x,y)² = (x-y)^T G (x-y).

**Parameters:**
- `x`: First embeddings, shape (batch_size, embedding_dim)
- `y`: Second embeddings, shape (batch_size, embedding_dim)  
- `squared`: If True, return squared distance

**Returns:** Distances of shape (batch_size,)

##### `compute_pairwise_distances(X, Y=None, squared=True) -> torch.Tensor`
Compute pairwise Mahalanobis distances between sets of embeddings.

**Parameters:**
- `X`: First set, shape (n, embedding_dim)
- `Y`: Second set, shape (m, embedding_dim). If None, compute within X
- `squared`: If True, return squared distances

**Returns:** Distance matrix of shape (n, m)

##### `compute_condition_number() -> float`
Compute condition number of metric tensor for monitoring.

**Returns:** Condition number of G

##### `clip_spectrum(min_eigenvalue=1e-6)`
Clip spectrum to prevent numerical issues. Modifies A in-place.

##### `get_eigenvalues() -> torch.Tensor`
Get sorted eigenvalues of the metric tensor.

##### `regularization_loss(target_condition=10.0) -> torch.Tensor`
Compute regularization loss to encourage well-conditioned metric.

##### `get_stats() -> Dict[str, float]`
Get comprehensive statistics about the metric tensor.

**Returns:** Dict with condition number, eigenvalue stats, etc.

**Example Usage:**
```python
from src import SPDMetric
import torch

# Create metric
metric = SPDMetric(embedding_dim=64, rank=16)

# Compute distances  
x = torch.randn(32, 64)
y = torch.randn(32, 64)
distances = metric(x, y)

# Monitor condition number
condition = metric.compute_condition_number()
print(f"Condition number: {condition:.2f}")

# Get detailed statistics
stats = metric.get_stats()
print(f"Stats: {stats}")
```

#### `BatchedSPDMetric`

Extension for efficient batched operations.

```python
class BatchedSPDMetric(SPDMetric):
    def compute_triplet_distances(anchors, positives, negatives) -> Tuple[torch.Tensor, torch.Tensor]
    def compute_all_pairs_distances(batch, labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

### `src.curved_attention`

#### `CurvedMultiHeadAttention`

Multi-head attention with SPD-weighted distance computation.

```python
class CurvedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12, 
        dropout: float = 0.1,
        geometry_mode: str = 'none',
        rank: int = 16,
        device: Optional[torch.device] = None
    )
```

**Parameters:**
- `embed_dim`: Total embedding dimension (768 for DistilBERT)
- `num_heads`: Number of attention heads (12 for DistilBERT)
- `dropout`: Dropout probability
- `geometry_mode`: 'none', 'shared', or 'per_head'
- `rank`: Rank for SPD factorization
- `device`: Device to place tensors on

**Methods:**

##### `forward(hidden_states, attention_mask=None, head_mask=None, output_attentions=False)`

Forward pass with SPD-weighted or standard attention.

**Parameters:**
- `hidden_states`: Input tensor, shape (batch_size, seq_len, embed_dim)
- `attention_mask`: Mask tensor, shape (batch_size, 1, 1, seq_len)  
- `head_mask`: Head mask tensor (not implemented)
- `output_attentions`: Whether to output attention weights

**Returns:** Tuple of (output, attention_weights)

##### `get_geometric_parameters() -> Dict[str, torch.Tensor]`
Get all geometric parameters for separate optimization.

##### `get_standard_parameters() -> Dict[str, torch.Tensor]`  
Get standard attention parameters.

##### `get_parameter_count() -> Dict[str, int]`
Get parameter counts matching paper's Table 3.

##### `get_condition_numbers() -> Dict[str, float]`
Get current condition numbers for all SPD metrics.

##### `clip_geometric_spectrum(min_eigenvalue=1e-6)`
Clip spectrum of all SPD metrics for numerical stability.

##### `get_attention_stats() -> Dict[str, float]`
Get attention statistics for monitoring.

**Example Usage:**
```python  
from src import CurvedMultiHeadAttention
import torch

# Create curved attention layer
attention = CurvedMultiHeadAttention(
    embed_dim=768,
    num_heads=12,
    geometry_mode='shared',
    rank=16
)

# Forward pass
batch_size, seq_len = 2, 128
hidden_states = torch.randn(batch_size, seq_len, 768)
outputs = attention(hidden_states)
output = outputs[0]  # (batch_size, seq_len, 768)

# Monitor parameters
param_counts = attention.get_parameter_count()
print(f"Geometric params: {param_counts['geometric']}")

# Check condition numbers
if attention.geometry_mode != 'none':
    conditions = attention.get_condition_numbers()
    print(f"Condition numbers: {conditions}")
```

#### `CurvedDistilBertAttention`

DistilBERT-compatible attention layer with SPD weighting.

```python
class CurvedDistilBertAttention(nn.Module):
    def __init__(self, config, geometry_mode='none', rank=16)
    def forward(query, key, value, mask, head_mask=None, output_attentions=False)
```

### `src.transformer_integration`

#### `CurvedDistilBertForSequenceClassification`

Complete DistilBERT model with selective curved attention layers.

**Factory Function:**
```python
def create_curved_distilbert(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    curved_layers: List[int] = [1, 2],
    geometry_mode: str = 'shared',
    rank: int = 16
) -> CurvedDistilBertForSequenceClassification
```

**Example Usage:**
```python
from src import create_curved_distilbert

# Create model with curved layers 1-2
model = create_curved_distilbert(
    model_name="distilbert-base-uncased",
    num_labels=2,
    curved_layers=[1, 2],
    geometry_mode='shared',
    rank=16
)

# Use like standard DistilBERT
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
```

### `src.training`

#### `TrainingConfig`

Configuration dataclass for training setup.

```python
@dataclass
class TrainingConfig:
    geometric_lr: float = 1e-4
    standard_lr: float = 1e-5
    patience: int = 3
    min_delta: float = 1e-4
    batch_size: int = 16
    eval_batch_size: int = 32
    max_memory_gb: float = 8.0
    max_condition_number: float = 1e6
    gradient_clip_norm: float = 1.0
    # ... additional fields
```

#### `DualOptimizerTrainer`

Dual optimizer trainer for separate geometric and standard parameter optimization.

```python
class DualOptimizerTrainer:
    def __init__(self, model, config, device)
    def train(train_dataloader, eval_dataloader, num_epochs) -> Dict[str, Any]
    def evaluate(eval_dataloader) -> Dict[str, float]
```

**Example Usage:**
```python
from src import DualOptimizerTrainer, TrainingConfig

config = TrainingConfig(
    geometric_lr=1e-4,
    standard_lr=1e-5,
    batch_size=16
)

trainer = DualOptimizerTrainer(model, config, device)
stats = trainer.train(train_loader, eval_loader, num_epochs=5)
```

### `src.evaluation`

#### `BootstrapAnalyzer`

Bootstrap confidence interval analysis.

```python
class BootstrapAnalyzer:
    def __init__(self, n_bootstrap=1000, confidence_level=0.95, seed=42)
    def bootstrap_correlation(x, y, method='spearman') -> Tuple[float, Tuple[float, float]]
    def bootstrap_difference(group1, group2) -> Tuple[float, Tuple[float, float]]
    def compute_effect_size(group1, group2) -> float
```

#### `StatisticalResult`

Container for statistical test results.

```python
@dataclass
class StatisticalResult:
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str
    significant: bool
```

### `src.synthetic_validation`

#### Core Functions

```python
def set_seed(seed: int) -> None
def create_synthetic_data(dimension, n_samples, true_eigenvalues, n_triplets, seed) -> Dict
def train_spd_metric(triplets, dimension, rank, learning_rate, n_epochs, margin) -> Dict  
def evaluate_eigenvalue_recovery(learned_metric, true_eigenvalues) -> Dict
def plot_eigenvalue_recovery(results, save_path=None) -> None
```

**Example Usage:**
```python
from src.synthetic_validation import *

# Create synthetic data
data = create_synthetic_data(
    dimension=10,
    n_samples=200, 
    true_eigenvalues=[5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
    n_triplets=1000,
    seed=42
)

# Train metric
results = train_spd_metric(
    triplets=data['triplets'],
    dimension=10,
    rank=10,
    learning_rate=0.01,
    n_epochs=100,
    margin=1.0
)

# Evaluate recovery
recovery = evaluate_eigenvalue_recovery(
    results['metric'],
    np.array([5, 3, 2, 1, 1, 1, 1, 1, 1, 1])
)

print(f"Correlation: {recovery['correlation']:.3f}")
```

## Utility Functions

### Device Management

```python
from src.curved_attention import validate_device_compatibility

# Check device compatibility
compat = validate_device_compatibility()
print(f"MPS working: {compat['mps_working']}")
print(f"CUDA working: {compat['cuda_working']}")
print(f"CPU working: {compat['cpu_working']}")
```

### Configuration Loading

```python
import yaml

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load experiment config
config = load_config('configs/phase4_config.yaml')
```

## Error Handling

### Common Exceptions

#### `SPDMetric` Errors
- `RuntimeError`: Numerical instability (condition number too high)
- `ValueError`: Invalid parameter combinations
- `torch.linalg.LinAlgError`: Matrix operation failures

#### `CurvedAttention` Errors  
- `AssertionError`: Invalid embed_dim/num_heads ratio
- `RuntimeError`: Device placement issues
- `ValueError`: Unknown geometry_mode

#### Training Errors
- `OutOfMemoryError`: Insufficient GPU/system memory
- `RuntimeError`: Gradient computation issues
- `ValueError`: Invalid batch dimensions

### Error Recovery

```python
try:
    metric = SPDMetric(embedding_dim=64, rank=16)
    distances = metric(x, y)
except RuntimeError as e:
    print(f"SPD metric error: {e}")
    # Fallback to Euclidean distance
    distances = torch.norm(x - y, dim=1)

try:
    condition = metric.compute_condition_number()
    if condition > 1e6:
        metric.clip_spectrum(min_eigenvalue=1e-6)
except Exception as e:
    print(f"Condition number error: {e}")
```

## Performance Tips

### Memory Optimization
```python
# Use smaller batch sizes
config.batch_size = 8
config.eval_batch_size = 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use CPU fallback for large models
device = torch.device('cpu')
```

### Speed Optimization
```python
# Use appropriate device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use DataLoader num_workers
dataloader = DataLoader(dataset, num_workers=4)
```

### Numerical Stability
```python
# Monitor condition numbers
if hasattr(model, 'get_condition_numbers'):
    conditions = model.get_condition_numbers()
    max_condition = max(conditions.values())
    if max_condition > 1e6:
        model.clip_geometric_spectrum()

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Enable autocast for mixed precision (optional)
with torch.autocast(device_type='cuda', enabled=False):  # Disabled for stability
    outputs = model(inputs)
```

## Integration Examples

### Custom Training Loop

```python
import torch
from src import SPDMetric, CurvedMultiHeadAttention

def custom_training_loop():
    # Setup
    metric = SPDMetric(embedding_dim=64, rank=16)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x, y, labels = batch
            
            # Compute distances
            distances = metric(x, y)
            
            # Triplet loss
            loss = torch.clamp(distances - margin, min=0).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
            optimizer.step()
            
            # Monitor stability
            condition = metric.compute_condition_number()
            if condition > 1e6:
                metric.clip_spectrum()
```

### Custom Attention Integration

```python
from transformers import DistilBertModel
from src import CurvedDistilBertAttention

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Replace specific attention layers
        self.bert.transformer.layer[1].attention = CurvedDistilBertAttention(
            config=self.bert.config,
            geometry_mode='shared',
            rank=16
        )
    
    def forward(self, **inputs):
        return self.bert(**inputs)
```

---

This API reference covers all public interfaces in the Geodesyxx reproduction package. For implementation details, see the source code in the `src/` directory.