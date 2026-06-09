import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";
import {elementwise_addition} from "../math/vector/sum";
import {ReLU} from "../math/relu";
import softmax from "../math/softmax";

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

    layers: Layer[] = []

    constructor(public readonly inputSize: number, public readonly hidden: number[], public readonly outputSize: number) {

        let currentInputs = inputSize;

        for (let i = 0; i < hidden.length; i++) {

            const layer = hidden[i]
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

        let acc: Vector = Vector.from(input);

        for (let i = 0; i < this.layers.length; i++) {

            const layer = this.layers[i];

            const mul_res = Matrix.matrixMulVector(layer.weight, acc)
            const z = Vector.from(elementwise_addition(mul_res, layer.bias.toVector()))

            if (i == this.layers.length - 1) {
                acc = Vector.from(softmax(z.toArray()))
            } else acc = ReLU(z);

        }

        return acc

    }

    backward() {

        const reversed_layers = this.layers.reverse();

        for (let i = 0; i < reversed_layers.length; i++) {
            const layer = reversed_layers[i];
        }

    }

}

