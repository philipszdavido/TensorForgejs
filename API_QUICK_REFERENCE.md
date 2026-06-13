# API Quick Reference

Quick lookup guide for TensorForgejs API.

## Core Data Structures

### Vector

```typescript
// Create
new Vector(size);
Vector.from([1, 2, 3]);
Vector.zeros(10);
Vector.random(10);

// Operations
v.set(index, value);
v.get(index);
v.length;
v.sum();
v.avg();
v.toArray();
v.print();
```

### Matrix

```typescript
// Create
new Matrix(rows, cols);
Matrix.zeros(rows, cols);

// Operations
m.set(r, c, value);
m.get(r, c);
m.getRow(r);
m.setRow(r, row);
(m.rows, m.columns);
m.print();

// Static methods
Matrix.matrixMulVector(m, v);
Matrix.outerProduct(v1, v2);
Matrix.add(m1, m2);
```

### Tensor

```typescript
new Tensor([1, 2, 3]);
Tensor.zeros(size);
t.clone();
t.data;
```

## Supervised Learning Models

### Linear Regression

```typescript
import { LinearRegression } from 'tensorforgejs';

const model = new LinearRegression(inputSize, bias);
model.predict(x)           // Prediction
model.forward(x)           // Forward pass
model.backward(x, y, yHat) // Backward pass
model.setWeights([...])
model.setBias(0.5)
```

### Logistic Regression

```typescript
import { LogisticRegression } from "tensorforgejs";

const model = new LogisticRegression(inputSize);
model.predict(x); // Returns probability [0, 1]
model.forward(x);
model.backward(x, y, yHat);
```

### K-Nearest Neighbors

```typescript
import { KNN } from "tensorforgejs";

const samples = [
  { data: [1, 2], label: "A" },
  { data: [8, 9], label: "B" },
];

const model = new KNN(samples, k);
model.predict(x); // Returns label
```

### SVM Models

```typescript
import {
  LinearSVM,
  PolynomialKernelSVM,
  SigmoidKernelSVM,
} from "tensorforgejs";

// Linear SVM
const linear = new LinearSVM(inputSize);

// Polynomial Kernel
const poly = new PolynomialKernelSVM(inputSize, degree);

// Sigmoid Kernel
const sigmoid = new SigmoidKernelSVM(inputSize);

// All SVM models support
model.forward(x);
model.backward(x, y, yHat);
```

### Discrete Perceptron

```typescript
import { DiscretePerceptron } from "tensorforgejs";

const model = new DiscretePerceptron(inputSize);
model.forward(x); // Returns 0 or 1
model.backward(x, y, yHat);
```

### Naive Bayes

```typescript
import { NaiveBayes } from "tensorforgejs";

const model = new NaiveBayes();
model.fit(features, labels);
model.predict(x);
```

### Polynomial Regression

```typescript
import { PolynomialRegression } from "tensorforgejs";

const model = new PolynomialRegression(inputSize, degree);
model.predict(x);
```

## Unsupervised Learning

### K-Means

```typescript
import { KMeans } from "tensorforgejs";

const model = new KMeans(k, maxIterations);
model.fit(features);
model.predict(x); // Returns cluster ID
```

## Neural Networks

### NeuralNetwork

```typescript
import { NeuralNetwork, ActivationEnum, MSELoss } from "tensorforgejs";

const model = new NeuralNetwork(
  { size: 784 }, // input
  [
    // hidden layers
    { size: 128, activation: ActivationEnum.ReLU },
    { size: 64, activation: ActivationEnum.ReLU },
  ],
  { size: 10, activation: ActivationEnum.Softmax }, // output
  new MSELoss(),
);

model.forward(x); // Forward pass
model.backward(y); // Backward pass
model.layers; // Access layers
```

### Activation Functions

```typescript
enum ActivationEnum {
  Linear = "linear",
  ReLU = "relu",
  Sigmoid = "sigmoid",
  Softmax = "softmax",
}
```

## Loss Functions

### MSE Loss

```typescript
import { MSELoss } from "tensorforgejs";

const loss = new MSELoss();
loss.compute(yTrue, yPred);
loss.gradient(yTrue, yPred);
```

### Cross Entropy Loss

```typescript
import { CrossEntropyLoss } from "tensorforgejs";

const loss = new CrossEntropyLoss();
```

### Softmax Cross Entropy

```typescript
import { SoftmaxCrossEntropy } from "tensorforgejs";

const loss = new SoftmaxCrossEntropy();
loss.forward(logits);
loss.fusedGradient(yTrue);
```

## Math Operations

### Activation Functions

```typescript
import { relu, sigmoid, softmax } from "tensorforgejs";

relu(vector);
sigmoid(vector);
softmax(vector);
```

### Matrix Operations

```typescript
import { transpose, matmul, reshape } from "tensorforgejs";

transpose(matrix);
matmul(m1, m2);
reshape(vector, shape);
```

### Utilities

```typescript
import {
  sum,
  arange,
  zeros,
  ones,
  gather,
  scatter,
  slice,
  euclidean_distance,
  minmax,
  standardize,
} from "tensorforgejs";

sum(vector);
arange(start, end);
zeros(size);
ones(size);
gather(vector, indices);
scatter(indices, values, shape);
slice(vector, start, end);
euclidean_distance(a, b);
minmax(vector);
standardize(vector);
```

## Model Inference

### ModelInferenceEngine

```typescript
import { ModelInferenceEngine } from 'tensorforgejs';

const modelData = {
  weights: [...],
  biases: [...],
  architecture: {...}
};

const engine = new ModelInferenceEngine(modelData);
engine.forward(input)  // Returns prediction
```

## Common Training Pattern

```typescript
const model = new LinearRegression(inputSize);
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch < epochs; epoch++) {
  for (const { x, y } of trainingData) {
    // Forward
    const yHat = model.predict(x);

    // Backward
    model.backward(x, y, yHat);

    // Update weights
    for (let i = 0; i < model.weights.length; i++) {
      model.weights[i] -= learningRate * model.gradWeights[i];
    }
    model.bias -= learningRate * model.gradBias;
  }
}
```

## Import Patterns

```typescript
// Import everything
import TensorForge from "tensorforgejs";
const { Core, Models, ModelInferenceEngine } = TensorForge;

// Import specific modules
import { Vector, Matrix, Tensor } from "tensorforgejs";
import { LinearRegression, LogisticRegression, KNN } from "tensorforgejs";

// Import utilities
import { relu, sigmoid, softmax } from "tensorforgejs";
import { MSELoss, CrossEntropyLoss } from "tensorforgejs";
```

## Property Access

```typescript
// Vector properties
v.length; // Size of vector
v.data; // Underlying Float64Array

// Matrix properties
m.rows; // Number of rows
m.columns; // Number of columns
m.data; // Underlying Float64Array

// Model properties (all supervised models)
model.weights; // Weight parameters
model.bias; // Bias parameter
model.gradWeights; // Weight gradients
model.gradBias; // Bias gradient

// NeuralNetwork properties
model.layers; // All layers
model.input; // Input configuration
model.hidden; // Hidden layers config
model.output; // Output layer config
model.loss; // Loss function
```

## Error Handling

```typescript
// Data validation
import assert from "tensorforgejs";

assert(samples.length > 0, "Samples cannot be empty");
assert(m.columns === v.length, `Shape mismatch: ${m.columns} !== ${v.length}`);
```

---

For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)
