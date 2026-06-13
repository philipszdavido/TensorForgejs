import {Matrix} from "../../core/Matrix";
import {Vector} from "../../core/Vector";
import {Activation, ActivationEnum, ActivationUse} from "./Types";

// this will have an input and output, activation
export default class DenseLayer {

    weight: Matrix // W (out × in)
    bias: Vector // b (out)
    dW: Matrix
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation

    activation: Activation;

    constructor(public readonly inputSize: number, public readonly outputSize: number, public readonly actEnum: ActivationEnum) {

        this.activation = ActivationUse[actEnum]

        this.weight = this.initializeWeights(
            outputSize,
            inputSize,
            this.activation
        );

        this.dW = Matrix.zeros(outputSize, inputSize);
        this.bias = Vector.zeros(outputSize);
        this.dB = Vector.zeros(outputSize);

    }

    forward(input: DenseLayer) {
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

}
