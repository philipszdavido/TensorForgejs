# Architecture Guide

Internal architecture and design patterns used in TensorForgejs.

## Project Structure

```
TensorForgejs/
├── src/
│   ├── index.ts                 # Main entry point
│   ├── api/
│   │   └── InferenceEngine.ts   # Model deployment
│   ├── assert/
│   │   └── assert.ts            # Assertion utilities
│   ├── core/
│   │   ├── index.ts
│   │   ├── Vector.ts            # 1D arrays
│   │   ├── Matrix.ts            # 2D arrays
│   │   └── Tensor.ts            # N-D arrays
│   ├── datasets/                # Data utilities
│   ├── error/
│   │   ├── error.ts             # Error functions
│   │   ├── MAE.ts               # Mean Absolute Error
│   │   ├── MSE.ts               # Mean Squared Error
│   │   ├── RMSE.ts              # Root Mean Squared Error
│   │   └── logerror.ts
│   ├── loss/
│   │   ├── index.ts
│   │   ├── BCELoss.ts           # Binary Cross Entropy
│   │   ├── cross_entropy.ts
│   │   ├── hinge.ts             # Hinge Loss (SVM)
│   │   └── MSELoss.ts
│   ├── math/
│   │   ├── arange.ts            # Range generation
│   │   ├── clip.ts              # Clipping values
│   │   ├── euclidean_distance.ts
│   │   ├── gather.ts            # Gather elements
│   │   ├── matmul.ts            # Matrix multiplication
│   │   ├── minmax.ts            # Min-max scaling
│   │   ├── ones.ts
│   │   ├── rand.ts
│   │   ├── random.ts
│   │   ├── randomNormal.ts      # Gaussian random
│   │   ├── relu.ts              # ReLU activation
│   │   ├── reshape.ts           # Reshape arrays
│   │   ├── scatter.ts           # Scatter elements
│   │   ├── sigmoid.ts           # Sigmoid activation
│   │   ├── slice.ts             # Array slicing
│   │   ├── softmax.ts           # Softmax activation
│   │   ├── standardize.ts       # Z-score normalization
│   │   ├── sum.ts               # Sum reduction
│   │   ├── transpose.ts         # Matrix transpose
│   │   ├── zeros.ts
│   │   └── vector/
│   │       ├── avg.ts           # Vector average
│   │       ├── dot.ts           # Dot product
│   │       └── sum.ts           # Vector sum
│   ├── models/
│   │   ├── index.ts
│   │   ├── Model.ts             # Base class
│   │   ├── KMeans.ts
│   │   ├── LinearRegression.ts
│   │   ├── LogisticRegression.ts
│   │   ├── NaiveBayes.ts
│   │   ├── Perceptron.ts
│   │   ├── PolyNomialRegression.ts
│   │   ├── knn/
│   │   │   ├── KNN.ts
│   │   │   └── WeightedKNN.ts
│   │   ├── neural/
│   │   │   ├── index.ts
│   │   │   ├── Activation.ts    # Activation definitions
│   │   │   ├── DenseLayer.ts    # Dense layer
│   │   │   ├── LossFunction.ts
│   │   │   ├── NeuralNetwork.ts
│   │   │   ├── SoftmaxCrossEntropy.ts
│   │   │   ├── Types.ts         # Type definitions
│   │   │   └── gpt/
│   │   │       ├── BPETokenizer.ts
│   │   │       ├── EmbeddingLayer.ts
│   │   │       ├── GPT.ts
│   │   │       ├── LayerNorm.ts
│   │   │       └── SelfAttention.ts
│   │   └── svm/
│   │       ├── KernelSVM.ts     # Base SVM
│   │       ├── LinearSVM.ts
│   │       ├── PolynomialKernelSVM.ts
│   │       └── SigmoidKernelSVM.ts
│   ├── optimizers/
│   │   ├── Optimizer.ts         # Base optimizer
│   │   └── SGD.ts               # Stochastic Gradient Descent
│   ├── regularizers/
│   │   ├── Lasso.ts             # L1 regularization
│   │   ├── Regularizer.ts       # Base class
│   │   └── Ridge.ts             # L2 regularization
│   └── trainers/
│       ├── Trainer.ts           # Base trainer
│       ├── LinearRegression.ts
│       ├── LinearSVMTrainer.ts
│       ├── PerceptronTrainer.ts
│       └── PolynomialRegressionTrainer.ts
├── test/                        # Test files
├── docs/                        # Existing documentation
├── dist/                        # Compiled output
├── package.json
├── tsconfig.json
└── README.md
```

## Design Patterns

### 1. Model Base Class Pattern

All supervised learning models inherit from a `Model` abstract base class:

```typescript
abstract class Model {
  protected weights: number[] = [];
  protected bias: number = 0;
  protected gradWeights: number[] = [];
  protected gradBias: number = 0;

  abstract forward(x: number[]): number;
  abstract backward(x: number[], y: number, yHat: number): void;
  abstract setBias(bias: number): void;
  abstract setWeights(weights: number[]): void;
}
```

**Benefits:**

- Consistent interface across all models
- Easy to swap models without changing training code
- Enforces implementation of required methods

**Example Implementations:**

- LinearRegression
- LogisticRegression
- Perceptron
- SVM variants

### 2. Activation Strategy Pattern

Activation functions are defined as objects with a consistent interface:

```typescript
type Activation = {
  forward(x: Vector): Vector;
  derivative(x: Vector, y: Vector): Vector;
  initializer(inputs: number): number;
};

export const ReLUActivation: Activation = {
  forward: ReLU,
  derivative: ReLU_derivative,
  initializer(inputs) {
    return Math.sqrt(2 / inputs); // He initialization
  },
};
```

**Benefits:**

- Easy to add new activation functions
- Consistent behavior across network layers
- Proper weight initialization per activation type

### 3. Typed Array Optimization

Core data structures use `Float64Array` for performance:

```typescript
export class Vector {
  private readonly data: Float64Array;

  constructor(size: number) {
    this.data = new Float64Array(size);
  }
}

export class Matrix {
  private readonly data: Float64Array;

  get(r: number, c: number): number {
    return this.data[r * this.columns + c]; // Row-major layout
  }
}
```

**Benefits:**

- Better memory locality
- Faster numerical operations
- ~8× memory efficiency vs JavaScript arrays
- Suitable for large-scale datasets

### 4. Loss Function Pattern

Loss functions support both computation and gradient calculation:

```typescript
interface LossFunction {
  compute(yTrue: Vector, yPred: Vector): number;
  gradient(yTrue: Vector, yPred: Vector): Vector;
}
```

**Fused Operations:**
SoftmaxCrossEntropy combines softmax and cross-entropy for numerical stability:

```typescript
export class SoftmaxCrossEntropy {
  forward(logits: Vector): Vector {
    // Numerically stable softmax
  }

  fusedGradient(yTrue: Vector): Vector {
    // Efficient gradient combining softmax + CE
    // This is more stable than computing separately
  }
}
```

### 5. Factory Pattern for Math Operations

Math utilities use factory and static methods:

```typescript
// Factory methods
export function relu(x: Vector): Vector { ... }
export function sigmoid(x: Vector): Vector { ... }

// Static methods on data structures
Matrix.zeros(rows, cols)
Vector.from(array)
```

## Data Flow

### Training Loop

```
User Code
    ↓
Model.forward(x) → Prediction
    ↓
Loss.compute(y, yHat) → Loss Value
    ↓
Model.backward(x, y, yHat) → Gradients
    ↓
Update Weights/Bias
    ↓
Loop to next batch
```

### Neural Network Forward Pass

```
Input Vector (784 dims)
    ↓
Dense Layer 1 (784 → 128)
├─ Matrix-Vector: weight @ input
├─ Add bias
├─ Apply ReLU activation
└─ Output: 128 dims
    ↓
Dense Layer 2 (128 → 64)
├─ Matrix-Vector: weight @ input
├─ Add bias
├─ Apply ReLU activation
└─ Output: 64 dims
    ↓
Dense Layer 3 (64 → 10)
├─ Matrix-Vector: weight @ input
├─ Add bias
├─ Apply Softmax activation
└─ Output: 10 dims (logits)
```

### Neural Network Backward Pass

```
Loss Gradient (output space)
    ↓
Layer 3 Backward
├─ Compute weight gradients
├─ Compute bias gradients
├─ Propagate to previous layer
└─ Output: gradients in 64-dim space
    ↓
Layer 2 Backward
├─ Apply activation derivative
├─ Compute weight gradients
├─ Propagate to previous layer
└─ Output: gradients in 128-dim space
    ↓
Layer 1 Backward
├─ Apply activation derivative
├─ Compute weight gradients
└─ Stop (input layer)
```

## Type System

### Core Types

```typescript
// Data structures
interface Vector {
  length: number;
  data: Float64Array;
}
interface Matrix {
  rows: number;
  columns: number;
  data: Float64Array;
}
interface Tensor {
  data: number[];
}

// ML Model types
type Model = {
  forward(x: number[]): number;
  backward(x: number[], y: number, yHat: number): void;
};

// Neural network types
type Layer = {
  weight: Matrix;
  bias: Vector;
  dW: Matrix;
  dB: Vector;
  activation: ActivationEnum;
  z?: Vector; // Pre-activation
  input?: Vector; // Layer input
  a?: Vector; // Activation output
};

type Input = { size: number };
type Hidden = { size: number; activation: ActivationEnum };
type Output = { size: number; activation: ActivationEnum };

// Functional types
type Loss = (...args: number[]) => number;
type Activation = {
  forward(x: Vector): Vector;
  derivative(x: Vector, y: Vector): Vector;
  initializer(inputs: number): number;
};

// Model persistence
type ModelPayload = {
  weights: number[][][];
  biases: number[][];
  architecture: {
    input: Input;
    hidden: Hidden[];
    output: Output;
  };
};
```

## Memory Management

### Memory Allocation Strategy

1. **Pre-allocate** all matrices/vectors at model creation
2. **Reuse** buffers within training loops
3. **Use typed arrays** for cache efficiency

Example:

```typescript
// BAD: Allocates new matrix every iteration
for (let i = 0; i < epochs; i++) {
  const grad = new Matrix(rows, cols); // ❌ Allocation overhead
}

// GOOD: Allocate once, reuse
const grad = Matrix.zeros(rows, cols);
for (let i = 0; i < epochs; i++) {
  grad.data.fill(0); // Reset to zero
  // Use grad...
}
```

### Memory Footprint

For a neural network with architecture [784 → 128 → 64 → 10]:

```
Layer 1 weights: 784 × 128 = 100,352 values × 8 bytes = 802 KB
Layer 1 bias:    128 values × 8 bytes = 1 KB
Layer 2 weights: 128 × 64 = 8,192 values × 8 bytes = 65 KB
Layer 2 bias:    64 values × 8 bytes = 512 B
Layer 3 weights: 64 × 10 = 640 values × 8 bytes = 5 KB
Layer 3 bias:    10 values × 8 bytes = 80 B

Total: ~874 KB per model instance

For gradients (same size): +874 KB
Total with gradients: ~1.7 MB
```

## Extension Points

### Adding a New Model

1. Extend `Model` base class:

```typescript
export class MyModel extends Model {
  constructor(inputSize: number, bias: number = 0) {
    super(inputSize, bias);
  }

  forward(x: number[]): number {
    // Implement forward pass
  }

  backward(x: number[], y: number, yHat: number): void {
    // Implement backward pass
  }

  setBias(bias: number): void {
    this.bias = bias;
  }

  setWeights(weights: number[]): void {
    this.weights = weights;
  }
}
```

2. Export from `src/models/index.ts`
3. Add usage example to documentation

### Adding a New Activation Function

1. Implement activation and derivative:

```typescript
export function myActivation(x: Vector): Vector {
  const result = new Vector(x.length);
  for (let i = 0; i < x.length; i++) {
    result.set(i /* activation logic */);
  }
  return result;
}

export function myActivation_derivative(x: Vector, y: Vector): Vector {
  const result = new Vector(y.length);
  for (let i = 0; i < y.length; i++) {
    result.set(i /* derivative logic */);
  }
  return result;
}
```

2. Create Activation object:

```typescript
export const MyActivation: Activation = {
  forward: myActivation,
  derivative: myActivation_derivative,
  initializer(inputs: number): number {
    return Math.sqrt(1 / inputs); // Xavier init
  },
};
```

3. Add to `ActivationEnum` and `ActivationUse` mapping

### Adding a New Math Operation

1. Implement the function:

```typescript
export default function myOperation(v: Vector): Vector {
  const result = new Vector(v.length);
  for (let i = 0; i < v.length; i++) {
    result.set(i /* operation logic */);
  }
  return result;
}
```

2. Export from appropriate module (`src/math/`, `src/math/vector/`, etc.)

3. Add to main `src/index.ts` export

## Performance Optimization Techniques

### 1. Loop Unrolling (Planned)

```typescript
// Standard loop
for (let i = 0; i < n; i++) {
  result += a[i] * b[i];
}

// Unrolled loop (4x)
for (let i = 0; i < n; i += 4) {
  result +=
    a[i] * b[i] +
    a[i + 1] * b[i + 1] +
    a[i + 2] * b[i + 2] +
    a[i + 3] * b[i + 3];
}
```

### 2. Cache Locality

Matrix stored in row-major order for better L1 cache hits during row operations.

### 3. SIMD-like Optimizations (Planned)

WebAssembly module for compute-intensive operations.

## Testing Strategy

### Unit Tests

- Test individual math operations
- Verify model predictions on toy datasets
- Test gradient calculations

### Integration Tests

- End-to-end training on standard datasets
- Verify convergence behavior
- Test model serialization/deserialization

### Benchmarks

- Performance profiling for matrix operations
- Memory usage tracking
- Comparison with baseline implementations

## Continuous Integration

Recommended CI pipeline:

```yaml
push → lint → build → test → benchmark → deploy-docs
```

## Contributing Guidelines

1. **Follow TypeScript conventions**
   - Use strict mode
   - Enable all compiler checks
   - Document public APIs

2. **Performance considerations**
   - Use typed arrays for numerical data
   - Avoid unnecessary allocations
   - Profile before optimizing

3. **Testing requirements**
   - Unit tests for new functions
   - Integration tests for model changes
   - Benchmark critical paths

4. **Documentation**
   - JSDoc comments for public APIs
   - Usage examples in docstrings
   - Update main documentation

---

See [DOCUMENTATION.md](DOCUMENTATION.md) for user-facing documentation.
