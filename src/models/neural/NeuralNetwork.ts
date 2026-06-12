import {Matrix} from "../../core/Matrix";
import {Vector} from "../../core/Vector";
import {elementwise_addition, elementwise_multiplication} from "../../math/vector/sum";
import transpose from "../../math/transpose";
import {SoftmaxCrossEntropy} from "./SoftmaxCrossEntropy";
import {Activation, ActivationEnum, ActivationUse, Hidden, Input, Layer, LossFunction, Output} from "./Types";

// This is currently Dense
// Will add:
// Conv2D
// Dropout
// BatchNorm
// Embedding
// Attention

export class NeuralNetwork {
    layers: Layer[] = [];
    private isTraining: boolean = true;

    constructor(
        public readonly input: Input,
        public readonly hidden: Hidden[],
        public readonly output: Output,
        public readonly loss: LossFunction | SoftmaxCrossEntropy
    ) {

        let currentInputs = input.size;

        for (let i = 0; i < hidden.length; i++) {

            const {size, activation} = hidden[i];

            const weight = this.initializeWeights(
                size,
                currentInputs,
                this.getActivation(activation)
            );

            const dW = Matrix.zeros(size, currentInputs);
            const bias = Vector.zeros(size);
            const dB = Vector.zeros(size);

            this.layers.push({weight, bias, dW, dB, activation});
            currentInputs = size;
        }

        const weight = this.initializeWeights(
            output.size,
            currentInputs,
            this.getActivation(output.activation)
        );

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

            layer.a = this.getActivation(layer.activation).forward(zWithBias)

            a = layer.a;

        }

        return a;
    }

    backward(y: number[]) {

        const output = this.layers[this.layers.length - 1];

        const Y: Vector = Vector.from(y);

        let delta!: Vector;

        if (this.loss instanceof SoftmaxCrossEntropy) {
            delta = this.loss.fusedGradient(Y);
        } else {

            // get the output error
            const Predicted = output.a!;
            const lossGrad = (this.loss as LossFunction).gradient(Y, Predicted);

            delta = Vector.from(
                elementwise_multiplication(lossGrad, this.getActivation(output.activation).derivative(output.z!, output.a!))
            );

        }

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];

            layer.dB = Vector.from(elementwise_addition(Vector.from(delta.toArray()), layer.dB));

            layer.dW = Matrix.add(layer.dW, Matrix.outerProduct(delta, layer!.input!));

            if (i > 0) {

                const prevLayer = this.layers[i - 1];
                const WT = transpose(layer.weight);

                delta = Vector.from(
                    elementwise_multiplication(
                        Matrix.matrixMulVector(WT, delta),
                        this.getActivation(prevLayer.activation).derivative(prevLayer.a!, prevLayer.z!)
                    ),
                );

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

                W.set(r, c, norm * std);
            }
        }

        return W;
    }

    public getWeightsAndBiases() {
        return {
            weights: this.layers.map(layer => layer.weight.toNestedArray()),
            biases: this.layers.map(layer => layer.bias.toArray())
        };
    }

    setMode(mode: 'train' | 'eval') {
        this.isTraining = mode === 'train';
    }

    getActivation(type: ActivationEnum) {
        return ActivationUse[type];
    }

}
