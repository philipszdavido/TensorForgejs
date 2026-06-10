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
};

export default class NeuralNetwork {
    layers: Layer[] = [];

    constructor(
        public readonly inputSize: number,
        public readonly hidden: number[],
        public readonly outputSize: number,
    ) {
        let currentInputs = inputSize;

        for (let i = 0; i < hidden.length; i++) {
            const layer = hidden[i];
            // outputs x inputs
            // new Matrix(cols, rows)
            const weight = Matrix.random(layer, currentInputs);
            const dW = Matrix.zeros(layer, currentInputs);
            const bias = Vector.random(layer);
            const dB = Vector.zeros(layer);

            this.layers.push({weight, bias, dW, dB});
            currentInputs = layer;
        }

        const weight = Matrix.random(outputSize, currentInputs);
        const bias = Vector.random(outputSize);
        const dW = Matrix.zeros(outputSize, currentInputs);
        const dB = Vector.zeros(outputSize);

        this.layers.push({weight, bias, dW, dB});
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
                layer.a = Vector.from(softmax(zWithBias.toArray()));
            } else {
                layer.a = ReLU(zWithBias);
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

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];

            error.print("Error");

            layer.weight.print("Weight");
            layer.input?.print("Input " + i);
            layer.z?.print("Z");
            layer.bias?.print("Bias " + i);

            layer.dB = Vector.from(error.toArray());
            layer.dB.print("dB " + i);

            layer.dW = Matrix.outerProduct(error, layer!.input!);

            layer.dW.print("dW " + i);

            if (i > 0) {
                const WT = transpose(layer.weight);

                WT.print("Transpose");
                error.print("Error");

                // error = Vector.vectorMulMatrix(error, layer.weight)
                error = Vector.from(
                    elementwise_multiplication(
                        Matrix.matrixMulVector(WT, error),
                        ReLU_derivative(this.layers[i - 1].z!),
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
                    const grad = layer.dW.get(r, c);
                    layer.weight.set(r, c, current - lr * grad);
                }
            }

            for (let j = 0; j < layer.bias.length; j++) {
                const current = layer.bias.get(j);
                const grad = layer.dB.get(j);

                layer.bias.set(j, current - lr * grad);
            }
        }
    }
}
