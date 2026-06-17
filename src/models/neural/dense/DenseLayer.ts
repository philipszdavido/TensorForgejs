import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {Activation, ActivationEnum, ActivationUse, LayerInterface} from "../Types";
import transpose from "../../../math/transpose";

// Table for Tensor
// | Variable | Shape      |
// | -------- | ---------- |
// | input    | (1 × in)   |
// | weight   | (in × out) |
// | bias     | (out)      |
// | output   | (1 × out)  |
// | dW       | (in × out) |
// | dB       | (out)      |

// this will have an input and output, activation
export default class DenseLayer implements LayerInterface {

    weight: Matrix // W (out × in)
    bias: Vector // b (out)
    dW: Matrix
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation

    activation: Activation;

    l1?: Vector; // L1 Regularization
    l2?: Vector; // L2 Regularization

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

    forward(input: number[]) {

        let a: Vector = Vector.from(input);

        const z = Matrix.matrixMulVector(this.weight, a);

        const zWithBias = Vector.addVectors(z, this.bias);

        this.z = zWithBias;
        this.input = a;

        this.a = this.activation.forward(zWithBias)

        a = this.a;

        return a;

    }

    backward(delta: Vector) {

        this.dB = Vector.addVectors(Vector.from(delta.toArray()), this.dB);

        this.dW = Matrix.add(this.dW, Matrix.outerProduct(delta, this!.input!));


        const WT = transpose(this.weight);
        return Matrix.matrixMulVector(WT, delta)

    }

    updateWeights(learningRate: number) {
        this.weight = Matrix.sub(this.weight, Matrix.multiplyScalar(this.dW, learningRate))
        this.dW = Matrix.zeros(this.dW.rows, this.dW.columns)

        this.bias = Vector.subVectors(this.bias, Vector.multiplyScalar(this.dB, learningRate))
        this.dB = Vector.zeros(this.dB.length)
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

    getWeight() {
        return this.weight.toNestedArray()
    }

    getBias() {
        return this.bias.toArray();
    }

    setWeight(weights: number[][]) {
        this.weight = Matrix.from(weights)
    }

    setBias(bias: number[]) {
        this.bias = Vector.from(bias)
    }

}
