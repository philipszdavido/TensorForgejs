import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";
import {elementwise_addition} from "../math/vector/sum";
import {ActivationEnum, ActivationUse} from "../models";

export type Activation = {
    forward(x: Vector): Vector;
    derivative(x: Vector, y: Vector): Vector;
    initializer(inputs: number): number;
};

export type Input = { size: number; };
export type Hidden = { size: number; activation: ActivationEnum; };
export type Output = { size: number; activation: ActivationEnum; };

export type Layer = {
    weight: Matrix;
    bias: Vector;
    activation: ActivationEnum;
};

export interface ModelPayload {
    weights: number[][][];
    biases: number[][];
    architecture: {
        input: Input;
        hidden: Hidden[];
        output: Output;
    };
}

export class ModelInferenceEngine {
    private layers: Layer[] = [];

    constructor(modelData: ModelPayload) {
        this.assembleLayers(modelData);
    }

    private assembleLayers(modelData: ModelPayload): void {
        const {weights, biases, architecture} = modelData;
        const totalLayers = architecture.hidden.length + 1;

        for (let i = 0; i < totalLayers; i++) {
            const isOutputLayer = i === architecture.hidden.length;

            const activationStrategy = isOutputLayer
                ? architecture.output.activation
                : architecture.hidden[i].activation;

            const layer: Layer = {
                weight: Matrix.from(weights[i]),
                bias: Vector.from(biases[i]),
                activation: activationStrategy
            };

            this.layers.push(layer);
        }
    }

    public forward(input: number[]) {

        let a: Vector = Vector.from(input);

        for (let i = 0; i < this.layers.length; i++) {

            const layer = this.layers[i];

            const z = Matrix.matrixMulVector(layer.weight, a);

            const zWithBias = Vector.from(elementwise_addition(z, layer.bias));

            a = ActivationUse[layer.activation].forward(zWithBias)

        }

        return a.toArray();
    }

}
