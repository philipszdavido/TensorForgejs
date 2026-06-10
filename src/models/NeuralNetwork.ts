import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";
import {elementwise_addition, elementwise_multiplication, vector_subtract} from "../math/vector/sum";
import {ReLU, ReLU_derivative} from "../math/relu";
import softmax from "../math/softmax";
import transpose from "../math/transpose";
import sigmoid, {Sigmoid, Sigmoid_derivative} from "../math/sigmoid";
import randomNormal from "../math/randomNormal";
import clip, {normClip} from "../math/clip";

// This is currently Dense
// Will add:
// Conv2D
// Dropout
// BatchNorm
// Embedding
// Attention

type LossFunction = {
    loss(y: Vector, pred: Vector): number;
    gradient(y: Vector, pred: Vector): Vector;
    fused?: boolean;
};

export const MSE: LossFunction = {

    loss(y, pred) {
        let sum = 0;

        for (let i = 0; i < y.length; i++) {
            const d = y.get(i) - pred.get(i);
            sum += d * d;
        }

        return sum / y.length;
    },

    gradient(y, pred) {
        return vector_subtract(pred, y);
    }
};

export const BCE: LossFunction = {
    loss: (y, pred) => -y.toArray().reduce((s, v, i) => {
        const p = Math.max(pred.get(i), 1e-15);
        return s + v * Math.log(p) + (1 - v) * Math.log(Math.max(1 - p, 1e-15));
    }, 0) / y.length,
    gradient: (y, pred) => Vector.from(pred.toArray().map((p, i) => {
        const t = y.get(i);
        const eps = 1e-15;
        return -(t / Math.max(p, eps) - (1 - t) / Math.max(1 - p, eps)) / y.length;
    }))
    // loss(y, pred) {
    //     let sum = 0;
    //     const eps = 1e-12;
    //
    //     for (let i = 0; i < y.length; i++) {
    //         const p = pred.get(i);
    //         const t = y.get(i);
    //
    //         sum += -(
    //             t * Math.log(p + eps) +
    //             (1 - t) * Math.log(1 - p + eps)
    //         );
    //     }
    //
    //     return sum;
    // },
    //
    // // _gradient(y, pred) {
    // //     return vector_subtract(pred, y);
    // // },
    // gradient(y, pred) {
    //     const gradData: number[] = [];
    //     const eps = 1e-12;
    //
    //     for (let i = 0; i < y.length; i++) {
    //         const p = pred.get(i);
    //         const t = y.get(i);
    //
    //         // const g = (p - t) / (p * (1 - p) + eps);
    //         const g = -(t / Math.max(p, 1e-15) - (1 - t) / Math.max(1 - p, 1e-15));
    //         gradData.push(g);
    //     }
    //     return Vector.from(gradData);
    // }
};

export const SoftmaxCrossEntropy: LossFunction = {
    fused: true,

    loss(y, pred) {
        let sum = 0;
        const eps = 1e-12;
        for (let i = 0; i < y.length; i++) {
            sum += -y.get(i) * Math.log(pred.get(i) + eps);
        }
        return sum;
    },

    gradient(y, pred) {
        return vector_subtract(pred, y);
    }
};

type Activation = {
    forward(x: Vector): Vector;
    derivative(x: Vector, y: Vector): Vector;
    initializer(inputs: number): number;
};

export const LinearActivation: Activation = {
    forward: (x) => x,
    derivative: (x) => {
        const data = new Array(x.length).fill(1);
        return Vector.from(data);
    },
    initializer(inputs) {
        return Math.sqrt(1 / inputs); // Xavier
    }
};

export const ReLUActivation: Activation = {
    forward: ReLU,
    derivative: ReLU_derivative,
    initializer(inputs) {
        return Math.sqrt(2 / inputs);
    }
};

export const SigmoidActivation: Activation = {
    forward: Sigmoid,
    derivative: Sigmoid_derivative,
    initializer(inputs) {
        return Math.sqrt(1 / inputs);
    }
};

export const SoftmaxActivation: Activation = {

    forward(x) {
        return Vector.from(
            softmax(x.toArray())
        );
    },

    derivative(x) {
        throw new Error(
            "Softmax derivative should be fused with CrossEntropy"
        );
    },

    initializer(inputs) {
        return Math.sqrt(1 / inputs);
    }
};

type Input = {
    size: number;
}

type Hidden = {
    size: number;
    activation: Activation;
}

type Output = {
    size: number;
    activation: Activation;
};

type Layer = {
    weight: Matrix, // W (out × in)
    bias: Vector, // b (out)
    dW: Matrix,
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation

    activation: Activation;

    activation: Activation;
};

export default class NeuralNetwork {
    layers: Layer[] = [];

    constructor(
        public readonly input: Input,
        public readonly hidden: Hidden[],
        public readonly output: Output,
        public readonly loss: LossFunction
    ) {

        if (
            output.activation === SoftmaxActivation &&
            !loss.fused
        ) {
            throw new Error(
                "Softmax requires SoftmaxCrossEntropy"
            );
        }

        let currentInputs = input.size;

        for (let i = 0; i < hidden.length; i++) {
            const {size, activation} = hidden[i];
            // outputs x inputs
            // new Matrix(cols, rows)
            const weight = this.initializeWeights(
                size,
                currentInputs,
                activation
            );
            // Matrix.random(size, currentInputs);

            const dW = Matrix.zeros(size, currentInputs);
            const bias = Vector.zeros(size);
            const dB = Vector.zeros(size);

            this.layers.push({weight, bias, dW, dB, activation});
            currentInputs = size;
        }

        const weight = this.initializeWeights(
            output.size,
            currentInputs,
            output.activation
        );
        // Matrix.random(output.size, currentInputs);
        const bias = Vector.zeros(output.size);
        const dW = Matrix.zeros(output.size, currentInputs);
        const dB = Vector.zeros(output.size);

        this.layers.push({weight, bias, dW, dB, activation: output.activation});
    }

    forward(input: number[]) {

        let a: Vector = Vector.from(input);

        for (let i = 0; i < this.layers.length; i++) {

            const layer = this.layers[i];

            const z = Matrix.matrixMulVector(layer.weight, a);

            const zWithBias = Vector.from(elementwise_addition(z, layer.bias));

            layer.z = zWithBias;
            layer.input = a;

            if (i === this.layers.length - 1) {
                layer.a = layer.activation.forward(zWithBias) // Vector.from(softmax(zWithBias.toArray()));
            } else {
                layer.a = layer.activation.forward(zWithBias) // ReLU(zWithBias);
            }

            a = layer.a;

        }

        return a;
    }

    backward(y: number[]) {
        console.log("==============Backward=================");

        const output = this.layers[this.layers.length - 1];
        const Predicted = output.a!;

        const Y: Vector = Vector.from(y);

        let error = vector_subtract(Predicted, Y);

        Y.print("Y");
        Predicted?.print("Predicted");

        // get the output error
        const lossGrad = this.loss.gradient(Y, Predicted);

        let delta: Vector;

        // if (this.loss.fused) {
        //     delta = lossGrad;
        // } else {
        delta = Vector.from(
            elementwise_multiplication(lossGrad, output.activation.derivative(output.z!, output.a!))
        );
        //}

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];

            layer.dB = Vector.from(elementwise_addition(Vector.from(delta.toArray()), layer.dB));

            layer.dW = Matrix.add(layer.dW, Matrix.outerProduct(delta, layer!.input!));
            layer.dW = Matrix.outerProduct(error, layer!.input!);

            if (i > 0) {

                const prevLayer = this.layers[i - 1];
            if (i > 0) {
                const WT = transpose(layer.weight);

                WT.print("Transpose");
                error.print("Error");

                // error = Vector.vectorMulMatrix(error, layer.weight)
                delta = Vector.from(
                    elementwise_multiplication(
                        Matrix.matrixMulVector(WT, delta),
                        prevLayer.activation.derivative(prevLayer.a!, prevLayer.z!) //ReLU_derivative(this.layers[i - 1].z!),
                    ),
                );

                error.print("Error");
            }
        }
    }

    update(lr: number) {
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            for (let r = 0; r < layer.weight.rows; r++) {
                for (let c = 0; c < layer.weight.columns; c++) {
                    const current = layer.weight.get(r, c);
                    const grad = (layer.dW.get(r, c));
                    layer.weight.set(r, c, current - lr * grad);
                    layer.dW.set(r, c, 0);
                }
            }

            for (let j = 0; j < layer.bias.length; j++) {
                const current = layer.bias.get(j);
                const grad = (layer.dB.get(j));

                layer.bias.set(j, current - lr * grad);
                layer.dB.set(j, 0);
            }
        }
    }

    initializeWeights(
        outputs: number,
        inputs: number,
        activation: Activation
    ): Matrix {

        const std = activation.initializer(inputs);

        const W = Matrix.zeros(outputs, inputs);

        for (let r = 0; r < outputs; r++) {
            for (let c = 0; c < inputs; c++) {

                const u1 = Math.random() || 1e-10;
                const u2 = Math.random();
                const norm = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

                W.set(
                    r,
                    c,
                    norm * std
                );
            }
        }

        return W;
    }

    setMode(mode: 'train' | 'eval') {
        this.isTraining = mode === 'train';
    }

}
