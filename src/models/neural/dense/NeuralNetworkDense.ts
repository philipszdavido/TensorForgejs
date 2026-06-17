import DenseLayer from "./DenseLayer";
import {Vector} from "../../../core/Vector";
import {SoftmaxCrossEntropy} from "../SoftmaxCrossEntropy";
import {LayerInterface, LossFunction} from "../Types";
import {Matrix} from "../../../core/Matrix";

export class NeuralNetworkDense implements LayerInterface {
    constructor(
        public readonly denseLayers: DenseLayer[],
        public readonly loss: LossFunction | SoftmaxCrossEntropy,
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

            delta = Vector.mulVectors(lossGrad, output.activation.derivative(output.z!, output.a!));

        }

        for (let i = this.denseLayers.length - 1; i >= 0; i--) {
            const denseLayer = this.denseLayers[i];
            const prevLayer = this.denseLayers[i - 1]
            const dA = denseLayer.backward(delta);

            if (i > 0) {
                delta = Vector.mulVectors(
                    dA,
                    prevLayer.activation.derivative(prevLayer.a!, prevLayer.z!)
                )
            } else delta = dA
        }

        return delta;

    }

    update(lr: number) {
        for (let i = 0; i < this.denseLayers.length; i++) {
            const denseLayer = this.denseLayers[i];
            denseLayer.updateWeights(lr)
        }
    }

    public applyBatchGradients(batchSize: number, learningRate: number) {
        for (const layer of this.denseLayers) {

            layer.dW = Matrix.multiplyScalar(layer.dW, 1 / batchSize);
            layer.dB = Vector.from(layer.dB.toArray().map(b => b / batchSize));

            layer.updateWeights(learningRate);
        }
    }

    lastDenseLayer() {
        return this.denseLayers[this.denseLayers.length - 1];
    }

    public getWeights() {
        return this.denseLayers.map((layer, index) => ({
            layerIndex: index,
            weights: layer.getWeight(),
            biases: layer.getBias()
        }));
    }

    public setWeights(savedLayers: any[]) {
        savedLayers.forEach((savedLayer, index) => {
            this.denseLayers[index].setWeight(savedLayer.weights);
            this.denseLayers[index].setBias(savedLayer.biases);
        });
    }

}
