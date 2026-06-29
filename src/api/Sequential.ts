import {ActivationEnum, LayerInterface, LossEnum, SoftmaxCE} from "../models";
import {Convo2D} from "../models/neural/convo2d/Convo2D";
import {Matrix} from "../core/Matrix";
import {ReLU2D} from "../models/neural/convo2d/ReLU2DLayer";
import {MaxPooling2D} from "../models/neural/convo2d/MaxPooling2D";
import {Flatten} from "../models/neural/convo2d/Flatten";
import {NeuralNetworkDense} from "../models/neural/dense/NeuralNetworkDense";
import DenseLayer from "../models/neural/dense/DenseLayer";

type Convo2DLayer = {
    type: "Convo2D",
    filters: number,
    kernelSize: number,
    stride: number,
    input: { rows: number, columns: number }
}

type ReLU2DLayer = {
    type: "ReLU2D",
}

type MaxPooling2DLayer = {
    type: "MaxPooling2D", filters: number, poolSize: number, stride: number
}

type FlattenLayer = {
    type: "Flatten",
}

type Dense = {
    type: "Dense", layers: [
        { inputSize: number, outputSize: number, activation: ActivationEnum },
        { inputSize: number, outputSize: number, activation: ActivationEnum }
    ]
}

export type SequentialLayer = Convo2DLayer | ReLU2DLayer | MaxPooling2DLayer | FlattenLayer | Dense;

export type ModelState = {
    weights: {
        convWeights: number[][][],
        denseWeights: { layerIndex: number, weights: number[][], biases: number[] }[],
    },
    architecture: Array<SequentialLayer>,
    embedding?: {
        data: Float32Array<ArrayBufferLike>,
        vocabSize: number,
        dim: number,
    },
    vocabularyMap?: {
        frequency: { [key: string]: number },
        vocab: { [key: string]: number },
    },
    loss: LossEnum
};

export class SequentialModel {
    private pipeline: LayerInterface[] = [];

    constructor(rawData: ModelState) {
        this.loadAndBuildModel(rawData);
    }

    private loadAndBuildModel(rawData: ModelState) {
        const {architecture, weights} = rawData;

        architecture.forEach((layerConf: SequentialLayer) => {
            switch (layerConf.type) {
                case 'Convo2D':
                    const convo2DFilters = new Array(layerConf.filters).fill(Matrix.random(layerConf.kernelSize, layerConf.kernelSize))

                    const convLayer = new Convo2D(convo2DFilters, layerConf.stride);
                    convLayer.setInputRowsAndColumns(layerConf.input.rows, layerConf.input.columns)

                    convLayer.setWeights(weights.convWeights, layerConf.kernelSize);
                    this.pipeline.push(convLayer);
                    break;

                case 'ReLU2D':
                    this.pipeline.push(new ReLU2D());
                    break;

                case 'MaxPooling2D':
                    const poolMap = Matrix.fromArray(new Array(layerConf.poolSize * layerConf.poolSize).fill(1), layerConf.poolSize, layerConf.poolSize);
                    const maxPooling2DFilters = new Array(layerConf.filters).fill(poolMap);
                    this.pipeline.push(new MaxPooling2D(maxPooling2DFilters, layerConf.stride));
                    break;

                case 'Flatten':
                    this.pipeline.push(new Flatten());
                    break;

                case 'Dense':
                    const denseLayers = layerConf.layers.map((dLayer: any) => {
                        return new DenseLayer(dLayer.inputSize, dLayer.outputSize, dLayer.activation);
                    });

                    const denseNet = new NeuralNetworkDense(denseLayers, SoftmaxCE);
                    denseNet.setWeights(weights.denseWeights);
                    this.pipeline.push(denseNet);
                    break;

                default:
                    throw new Error(`Unknown layer type in config: ${(layerConf as SequentialLayer).type}`);
            }
        });
        console.log(`Successfully built model with ${this.pipeline.length} layers dynamic pipeline.`);
    }

    public predict(inputData: any): number[] {
        let currentOutput = inputData;

        for (const layer of this.pipeline) {

            if (layer instanceof NeuralNetworkDense) {
                currentOutput = layer.forward(currentOutput?.toArray ? currentOutput.toArray() : currentOutput);
            } else {
                if (layer instanceof Convo2D) {
                    const {rows, columns} = layer.getInputRowsColumns()
                    currentOutput = Matrix.fromArray(currentOutput, rows, columns)
                }
                currentOutput = layer.forward(currentOutput);
            }
        }

        return currentOutput.toArray();
    }

    public getPipeLine(index: number) {
        return this.pipeline[index];
    }

}
