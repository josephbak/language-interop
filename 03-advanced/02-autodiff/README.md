# 02-autodiff: Automatic Differentiation

Forward-mode automatic differentiation using dual numbers.

## Concept

Dual numbers carry both value and derivative:
```
x = (value, derivative)
```

Operations propagate derivatives automatically:
```
(a, a') + (b, b') = (a + b, a' + b')           # sum rule
(a, a') * (b, b') = (a*b, a'*b + a*b')         # product rule
sin(a, a') = (sin(a), cos(a) * a')             # chain rule
```

## Usage
```python
import autodiff as ad

# Create a variable (seeded with grad=1)
x = ad.var(2.0)

# Build expression
f = x * x + 3 * x  # f(x) = x² + 3x

# Get value and derivative
print(f.val)   # 10 (f(2) = 4 + 6)
print(f.grad)  # 7  (f'(2) = 2*2 + 3)
```

## Supported Operations

| Operation | Derivative |
|-----------|------------|
| `a + b` | `a' + b'` |
| `a - b` | `a' - b'` |
| `a * b` | `a'*b + a*b'` |
| `a / b` | `(a'*b - a*b') / b²` |
| `sin(a)` | `cos(a) * a'` |
| `cos(a)` | `-sin(a) * a'` |
| `exp(a)` | `exp(a) * a'` |
| `log(a)` | `a' / a` |
| `pow(a, n)` | `n * a^(n-1) * a'` |
| `sqrt(a)` | `a' / (2 * sqrt(a))` |

## Forward vs Reverse Mode

This implements **forward mode** — good for few inputs, many outputs.

**Reverse mode** (backpropagation) is better for many inputs, few outputs (neural networks). That's what PyTorch/TensorFlow use.

## Build and Run

```bash
make run
```