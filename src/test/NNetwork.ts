import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";
import relu, {ReLU_derivative} from "../math/relu";
import {Sigmoid, Sigmoid_derivative} from "../math/sigmoid";

export type ActivationName = "sigmoid" | "relu" | "leaky_relu" | "tanh" | "linear" | "swish" | "elu";
export type LossName = "mse" | "mae" | "cross_entropy" | "binary_cross_entropy";
export type OptimizerName = "sgd" | "adam" | "rmsprop" | "momentum";

export interface LayerConfig {
    units: number;
    activation?: ActivationName;
    l1?: number;
    l2?: number;
}

export interface NeuralNetworkConfig {
    inputSize: number;
    layers: LayerConfig[];
    loss?: LossName;
    optimizer?: OptimizerName;
    learningRate?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
    momentum?: number;
    rmsDecay?: number;
}

type Activation = {
    forward(x: Vector): Vector;
    derivative(z: Vector, a: Vector): Vector; // Accepts both pre & post evaluations for optimization stability
    initScale(inputs: number, outputs: number): number;
};

export const Activations: Record<ActivationName, Activation> = {
    sigmoid: {
        forward: Sigmoid, // (x) => Vector.from(x.toArray().map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v)))))),
        derivative: Sigmoid_derivative, // (_, a) => Vector.from(a.toArray().map(s => s * (1 - s))),
        initScale: (in_) => Math.sqrt(1 / in_)
    },
    relu: {
        forward: (x) => Vector.from(x.toArray().map(v => relu(v))),
        derivative: ReLU_derivative, //(z, _) => Vector.from(z.toArray().map(v => (v > 0 ? 1 : 0))),
        initScale: (in_) => Math.sqrt(2 / in_)
    },
    leaky_relu: {
        forward: (x) => Vector.from(x.toArray().map(v => (v > 0 ? v : 0.01 * v))),
        derivative: (z, _) => Vector.from(z.toArray().map(v => (v > 0 ? 1 : 0.01))),
        initScale: (in_) => Math.sqrt(2 / in_)
    },
    tanh: {
        forward: (x) => Vector.from(x.toArray().map(v => Math.tanh(v))),
        derivative: (_, a) => Vector.from(a.toArray().map(v => 1 - v ** 2)),
        initScale: (in_, out_) => Math.sqrt(2 / (in_ + out_))
    },
    linear: {
        forward: (x) => x,
        derivative: (z, _) => Vector.from(new Array(z.length).fill(1)),
        initScale: (in_, out_) => Math.sqrt(2 / (in_ + out_))
    },
    swish: {
        forward: (x) => Vector.from(x.toArray().map(v => v / (1 + Math.exp(-v)))),
        derivative: (z, _) => Vector.from(z.toArray().map(v => {
            const sig = 1 / (1 + Math.exp(-v));
            return sig + v * sig * (1 - sig);
        })),
        initScale: (in_) => Math.sqrt(2 / in_)
    },
    elu: {
        forward: (x) => Vector.from(x.toArray().map(v => (v >= 0 ? v : Math.exp(v) - 1))),
        derivative: (z, _) => Vector.from(z.toArray().map(v => (v >= 0 ? 1 : Math.exp(v)))),
        initScale: (in_) => Math.sqrt(2 / in_)
    }
};

type LossFunction = {
    loss(y: Vector, pred: Vector): number;
    gradient(y: Vector, pred: Vector): Vector;
};

export const Losses: Record<LossName, LossFunction> = {
    mse: {
        loss: (y, pred) => y.toArray().reduce((s, v, i) => s + (pred.get(i) - v) ** 2, 0) / y.length,
        gradient: (y, pred) => Vector.from(pred.toArray().map((p, i) => (p - y.get(i)) / y.length))
    },
    mae: {
        loss: (y, pred) => y.toArray().reduce((s, v, i) => s + Math.abs(pred.get(i) - v), 0) / y.length,
        gradient: (y, pred) => Vector.from(pred.toArray().map((p, i) => {
            const act = y.get(i);
            return (p > act ? 1 : p < act ? -1 : 0) / y.length;
        }))
    },
    cross_entropy: {
        loss: (y, pred) => -y.toArray().reduce((s, v, i) => s + v * Math.log(Math.max(pred.get(i), 1e-15)), 0),
        gradient: (y, pred) => Vector.from(pred.toArray().map((p, i) => -y.get(i) / Math.max(p, 1e-15)))
    },
    binary_cross_entropy: {
        loss: (y, pred) => -y.toArray().reduce((s, v, i) => {
            const p = Math.max(pred.get(i), 1e-15);
            return s + v * Math.log(p) + (1 - v) * Math.log(Math.max(1 - p, 1e-15));
        }, 0) / y.length,
        gradient: (y, pred) => Vector.from(pred.toArray().map((p, i) => {
            const t = y.get(i);
            const eps = 1e-15;
            return -(t / Math.max(p, eps) - (1 - t) / Math.max(1 - p, eps)) / y.length;
        }))
    }
};

type Layer = {
    weight: Matrix;
    bias: Vector;
    dW: Matrix;
    dB: Vector;

    mW: Matrix; vW: Matrix; velW: Matrix; rmsW: Matrix;
    mB: Vector; vB: Vector; velB: Vector; rmsB: Vector;

    input?: Vector;
    z?: Vector;
    a?: Vector;

    activation: Activation;
    l1: number;
    l2: number;
};

export default class NeuralNetwork {
    readonly layers: Layer[] = [];
    private config: Required<NeuralNetworkConfig>;
    private step = 0;
    private isTraining = true;

    constructor(config: NeuralNetworkConfig) {
        this.config = {
            loss: "mse",
            optimizer: "adam",
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.9,
            rmsDecay: 0.9,
            ...config
        };

        let currentInputs = this.config.inputSize;

        for (const layerCfg of this.config.layers) {
            const outSize = layerCfg.units;
            const actName = layerCfg.activation || "relu";
            const activation = Activations[actName];

            const weight = Matrix.zeros(outSize, currentInputs);
            const scale = activation.initScale(currentInputs, outSize);

            for (let r = 0; r < outSize; r++) {
                for (let c = 0; c < currentInputs; c++) {
                    const u1 = Math.random() || 1e-10;
                    const u2 = Math.random();
                    const norm = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                    weight.set(r, c, norm * scale);
                }
            }

            this.layers.push({
                weight,
                bias: Vector.zeros(outSize),
                dW: Matrix.zeros(outSize, currentInputs),
                dB: Vector.zeros(outSize),
                mW: Matrix.zeros(outSize, currentInputs),
                vW: Matrix.zeros(outSize, currentInputs),
                velW: Matrix.zeros(outSize, currentInputs),
                rmsW: Matrix.zeros(outSize, currentInputs),
                mB: Vector.zeros(outSize),
                vB: Vector.zeros(outSize),
                velB: Vector.zeros(outSize),
                rmsB: Vector.zeros(outSize),
                activation,
                l1: layerCfg.l1 || 0,
                l2: layerCfg.l2 || 0
            });

            currentInputs = outSize;
        }
    }

    forward(input: number[] | Vector): Vector {
        let a = input instanceof Vector ? input : Vector.from(input);

        for (const layer of this.layers) {
            const z = Matrix.matrixMulVector(layer.weight, a);

            const zWithBias = new Vector(layer.bias.length);
            for (let j = 0; j < zWithBias.length; j++) {
                zWithBias.set(j, z.get(j) + layer.bias.get(j));
            }

            layer.input = a;
            layer.z = zWithBias;
            layer.a = layer.activation.forward(zWithBias);
            a = layer.a;
        }

        return a;
    }

    backward(y: number[] | Vector): void {
        const outputLayer = this.layers[this.layers.length - 1];
        if (!outputLayer.a || !outputLayer.z) {
            throw new Error("Execute forward pass operations before executing a backpropagation pass.");
        }

        const Y = y instanceof Vector ? y : Vector.from(y);
        const lossGrad = Losses[this.config.loss].gradient(Y, outputLayer.a);

        // delta = lossGrad ⊙ activation'(z)
        let delta = new Vector(lossGrad.length);
        const outDeriv = outputLayer.activation.derivative(outputLayer.z, outputLayer.a);
        for (let j = 0; j < delta.length; j++) {
            delta.set(j, lossGrad.get(j) * outDeriv.get(j));
        }

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];

            // Accumulate bias gradients (dB += delta)
            for (let j = 0; j < layer.bias.length; j++) {
                layer.dB.set(j, layer.dB.get(j) + delta.get(j));
            }

            // Accumulate weight gradients with L1/L2 updates: dW += (delta ⊗ input)
            for (let r = 0; r < layer.weight.rows; r++) {
                const dVal = delta.get(r);
                for (let c = 0; c < layer.weight.columns; c++) {
                    let grad = dVal * layer.input!.get(c);

                    if (layer.l1 > 0) grad += layer.l1 * Math.sign(layer.weight.get(r, c));
                    if (layer.l2 > 0) grad += layer.l2 * layer.weight.get(r, c);

                    layer.dW.set(r, c, layer.dW.get(r, c) + grad);
                }
            }

            // Route delta back into previous hidden layers (Transpose WT inline)
            if (i > 0) {
                const prevLayer = this.layers[i - 1];
                const nextDelta = new Vector(prevLayer.bias.length);

                for (let c = 0; c < layer.weight.columns; c++) {
                    let sum = 0;
                    for (let r = 0; r < layer.weight.rows; r++) {
                        sum += layer.weight.get(r, c) * delta.get(r);
                    }
                    nextDelta.set(c, sum);
                }

                const prevDeriv = prevLayer.activation.derivative(prevLayer.z!, prevLayer.a!);
                for (let j = 0; j < nextDelta.length; j++) {
                    nextDelta.set(j, nextDelta.get(j) * prevDeriv.get(j));
                }

                delta = nextDelta;
            }
        }
    }

    update(lr: number, batchSize = 1): void {
        this.step++;
        const {optimizer, beta1, beta2, epsilon, momentum, rmsDecay} = this.config;

        for (const layer of this.layers) {
            // Weights Updates
            for (let r = 0; r < layer.weight.rows; r++) {
                for (let c = 0; c < layer.weight.columns; c++) {
                    const g = layer.dW.get(r, c) / batchSize;
                    const wVal = layer.weight.get(r, c);

                    switch (optimizer) {
                        case "adam": {
                            const m = beta1 * layer.mW.get(r, c) + (1 - beta1) * g;
                            const v = beta2 * layer.vW.get(r, c) + (1 - beta2) * g * g;
                            layer.mW.set(r, c, m);
                            layer.vW.set(r, c, v);

                            const mHat = m / (1 - Math.pow(beta1, this.step));
                            const vHat = v / (1 - Math.pow(beta2, this.step));
                            layer.weight.set(r, c, wVal - (lr * mHat) / (Math.sqrt(vHat) + epsilon));
                            break;
                        }
                        case "momentum": {
                            const vel = momentum * layer.velW.get(r, c) + lr * g;
                            layer.velW.set(r, c, vel);
                            layer.weight.set(r, c, wVal - vel);
                            break;
                        }
                        case "rmsprop": {
                            const rms = rmsDecay * layer.rmsW.get(r, c) + (1 - rmsDecay) * g * g;
                            layer.rmsW.set(r, c, rms);
                            layer.weight.set(r, c, wVal - (lr * g) / (Math.sqrt(rms) + epsilon));
                            break;
                        }
                        default: // sgd
                            console.log("sgd")
                            layer.weight.set(r, c, wVal - lr * g);
                    }
                    layer.dW.set(r, c, 0); // Flush gradients
                }
            }

            // Biases Updates
            for (let j = 0; j < layer.bias.length; j++) {
                const g = layer.dB.get(j) / batchSize;
                const bVal = layer.bias.get(j);

                switch (optimizer) {
                    case "adam": {
                        const m = beta1 * layer.mB.get(j) + (1 - beta1) * g;
                        const v = beta2 * layer.vB.get(j) + (1 - beta2) * g * g;
                        layer.mB.set(j, m);
                        layer.vB.set(j, v);

                        const mHat = m / (1 - Math.pow(beta1, this.step));
                        const vHat = v / (1 - Math.pow(beta2, this.step));
                        layer.bias.set(j, bVal - (lr * mHat) / (Math.sqrt(vHat) + epsilon));
                        break;
                    }
                    case "momentum": {
                        const vel = momentum * layer.velB.get(j) + lr * g;
                        layer.velB.set(j, vel);
                        layer.bias.set(j, bVal - vel);
                        break;
                    }
                    case "rmsprop": {
                        const rms = rmsDecay * layer.rmsB.get(j) + (1 - rmsDecay) * g * g;
                        layer.rmsB.set(j, rms);
                        layer.bias.set(j, bVal - (lr * g) / (Math.sqrt(rms) + epsilon));
                        break;
                    }
                    default:
                        layer.bias.set(j, bVal - lr * g);
                }
                layer.dB.set(j, 0); // Flush gradients
            }
        }
    }

    trainStep(x: number[], y: number[], lr: number): number {
        const pred = this.forward(x);
        const Y = Vector.from(y);
        const lossValue = Losses[this.config.loss].loss(Y, pred);
        this.backward(y);
        this.update(lr, 1);
        return lossValue;
    }

    predict(input: number[]): number[] {
        return this.forward(input).toArray();
    }

    setMode(mode: "train" | "eval"): void {
        this.isTraining = mode === "train";
    }
}
