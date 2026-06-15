import {activate, activateDerivative} from "./activation";
import {Tensor} from "../../../core/Tensor";
import {NLayer} from "../Types";

type Activation = "relu" | "sigmoid" | "tanh" | "linear";

export class DenseLayer implements NLayer {
    private W: Tensor;
    private b: Tensor;

    private input!: Tensor;
    private z!: Tensor;

    private dW: Tensor;
    private dB: Tensor;

    constructor(
        private inputSize: number,
        private outputSize: number,
        private activation: Activation = "relu",
        private lr = 0.01
    ) {
        this.W = Tensor.from2DArray(
            Array.from({length: outputSize}, () =>
                Array.from({length: inputSize}, () => (Math.random() - 0.5) * 0.1)
            )
        );

        this.dW = Tensor.from2DArray(Array.from({length: outputSize}, () =>
            Array.from({length: inputSize}, () => 0)
        ))

        this.b = Tensor.fromArray(
            Array.from({length: outputSize}, () => 0)
        );

        this.dB = Tensor.fromArray(
            Array.from({length: outputSize}, () => 0)
        );

    }

    forward(input: Tensor): Tensor {
        this.input = input;

        const z = this.W.matMul(input).add(this.b);
        this.z = z;

        const activated = Tensor.fromArray(
            z.toArray().map(v => activate(v, this.activation))
        );

        return activated;
    }

    backward(grad: Tensor): Tensor {

        const dz = grad.toArray().map((g, i) =>
            g * activateDerivative(this.z.get(i), this.activation)
        );

        const inputArr = this.input.toArray();

        this.dW = Tensor.from2DArray(
            dz.map(d => inputArr.map(x => d * x))
        );

        this.dB = Tensor.fromArray(dz);

        // this.W = this.W.sub(dW.mulScalar(this.lr));
        // this.b = this.b.sub(db.mulScalar(this.lr));

        const W_T = this.W.transpose();

        return W_T.matMul(Tensor.fromArray(dz));
    }

    updateWeights(): void {

        this.W = this.W.sub(this.dW.mulScalar(this.lr));
        this.b = this.b.sub(this.dB.mulScalar(this.lr));

        this.dB = Tensor.zeros(this.dB.shape)
        this.dW = Tensor.zeros(this.dW.shape)

    }

    gradients(): Tensor[] {
        return [this.dW, this.dB];
    }

    parameters(): Tensor[] {
        return [];
    }

}
