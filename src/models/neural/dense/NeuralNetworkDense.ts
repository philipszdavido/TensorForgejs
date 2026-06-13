import DenseLayer from "../DenseLayer";
import {Vector} from "../../../core/Vector";
import {SoftmaxCrossEntropy} from "../SoftmaxCrossEntropy";
import {LossFunction} from "../Types";
import {elementwise_multiplication} from "../../../math/vector/sum";

export class NeuralNetworkDense {
    constructor(
        public readonly denseLayers: DenseLayer[],
        public readonly loss: LossFunction,
    ) {
    }

    forward(input: number[]) {

        let a: Vector = Vector.from(input);

        for (let i = 0; i < this.denseLayers.length; i++) {
            const denseLayer = this.denseLayers[i];
            a = denseLayer.forward(a.toArray());
        }

        return a;

    }

    backward(y: number[]) {

        const output = this.lastDenseLayer();

        const Y: Vector = Vector.from(y);

        let delta!: Vector;

        if (this.loss instanceof SoftmaxCrossEntropy) {
            delta = this.loss.fusedGradient(Y);
        } else {

            // get the output error
            const Predicted = output.a!;
            const lossGrad = (this.loss as LossFunction).gradient(Y, Predicted);

            delta = Vector.from(
                elementwise_multiplication(lossGrad, output.activation.derivative(output.z!, output.a!))
            );

        }

        for (let i = this.denseLayers.length - 1; i >= 0; i--) {
            const denseLayer = this.denseLayers[i];
            const prevLayer = this.denseLayers[i - 1]
            const dA = denseLayer.backward(delta);

            if (i > 0) {
                delta = Vector.from(
                    elementwise_multiplication(
                        dA,
                        prevLayer.activation.derivative(prevLayer.a!, prevLayer.z!)
                    ),
                );
            }
        }

    }

    update(lr: number) {
        for (let i = 0; i < this.denseLayers.length; i++) {
            const denseLayer = this.denseLayers[i];
            denseLayer.updateWeights(lr)
        }
    }

    lastDenseLayer() {
        return this.denseLayers[this.denseLayers.length - 1];
    }

}
