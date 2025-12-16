# 03-reverse-autodiff: Reverse-Mode Automatic Differentiation

Backpropagation—how neural networks compute gradients.

## Key Difference from Forward Mode

| | Forward Mode | Reverse Mode |
|--|--------------|--------------|
| Strategy | Compute derivatives during forward pass | Build graph, then backpropagate |
| Memory | Low | High (stores graph) |
| Efficiency | One pass per input | One pass for ALL inputs |
| Use case | Few inputs, many outputs | Many inputs, few outputs (neural networks) |

## How It Works

### Forward Pass: Build Graph
```python
x = Var(2.0)
y = Var(3.0)
f = (x + y) * x
```

Creates computation graph:
```
x ──┬──► (+) ──► a ──┐
    │                │
y ──┘                ├──► (*) ──► f
                     │
x ───────────────────┘
```

Each node stores:
- Value
- Backward edges (how to propagate gradients)

### Backward Pass: Propagate Gradients
```python
f.backward()  # computes ∂f/∂x, ∂f/∂y, etc.
```

1. Seed output: `f.grad = 1.0`
2. Traverse graph in reverse topological order
3. For each node, call gradient functions
4. Accumulate gradients at inputs

## Usage
```python
import autodiff as ad

# Create variables (inputs that need gradients)
x = ad.Var(2.0)
y = ad.Var(3.0)

# Build expression (forward pass)
f = (x + y) * x  # f = 10

# Backpropagate (backward pass)
f.backward()

# Access gradients
print(x.grad)  # ∂f/∂x = 7
print(y.grad)  # ∂f/∂y = 2
```

## Why This Powers Neural Networks
```python
# Simplified neural network training
w = ad.Var(0.5)  # weight
b = ad.Var(0.0)  # bias

for epoch in range(100):
    # Forward pass
    pred = x * w + b
    loss = (pred - y_true) ** 2
    
    # Backward pass
    loss.backward()
    
    # Update weights
    w.val -= learning_rate * w.grad
    b.val -= learning_rate * b.grad
    
    # Zero gradients for next iteration
    loss.zero_grad()
```

This is exactly what PyTorch does internally!

## Build and Run

```bash
make run
```