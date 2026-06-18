import {loadLocalMNIST} from "./mnistLoader";
import {Matrix} from "../src/core/Matrix";
import {Vector} from "../src/core/Vector";
import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
import {MaxPooling2D} from "../src/models/neural/convo2d/MaxPooling2D";
import {ReLU2D} from "../src/models/neural/convo2d/ReLU2DLayer";
import {ActivationEnum, LayerInterface, LossEnum, SoftmaxCE} from "../src/models";
import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {Flatten} from "../src/models/neural/convo2d/Flatten";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
import * as fs from 'fs';
import {ModelState} from "../src/api/Sequential";

const mnistData = loadLocalMNIST();

const SUBSET_SIZE = 1000;
const trainImagesFlat = mnistData.train.images//.slice(0, SUBSET_SIZE);
const trainLabels = mnistData.train.labels//.slice(0, SUBSET_SIZE);

function toOneHot(label: number, numClasses: number = 10): number[] {
    const oneHot = new Array(numClasses).fill(0.0);
    oneHot[label] = 1.0;
    return oneHot;
}

const FLATTENED_SIZE = 288//169;

const filters = [Matrix.random(4, 4), Matrix.random(4, 4)]
const conv = new Convo2D(filters, 1);

const relu2d = new ReLU2D();

const maxPoolFilters = [Matrix.fromArray([1, 1, 1, 1], 2, 2), Matrix.fromArray([1, 1, 1, 1], 2, 2)]
const pool = new MaxPooling2D(maxPoolFilters, 2);

const flatten = new Flatten();

const dense = new NeuralNetworkDense([
    new DenseLayer(FLATTENED_SIZE, 64, ActivationEnum.relu),
    new DenseLayer(64, 10, ActivationEnum.softmax)
], SoftmaxCE);

const layers: LayerInterface[] = [conv, relu2d, pool, flatten];

function trainMNIST(epochs: number, lr: number) {
    console.log(`--- Starting MNIST Training (${epochs} epochs) ---`);

    for (let e = 0; e < epochs; e++) {
        let epochLoss = 0;
        let correctPredictions = 0;

        for (let i = 0; i < trainImagesFlat.length; i++) {

            const inputMatrix = Matrix.fromArray(trainImagesFlat[i], 28, 28);

            const targetLabel = trainLabels[i];
            const targetVectorArray = toOneHot(targetLabel, 10);

            let out: any = conv.forward(inputMatrix);
            out = relu2d.forward(out);
            out = pool.forward(out);
            out = flatten.forward(out);
            const finalOut = dense.forward(out.toArray());

            epochLoss += SoftmaxCE.loss(Vector.from(targetVectorArray), finalOut);

            const probs = finalOut.toArray();
            const predictedClass = probs.indexOf(Math.max(...probs));
            if (predictedClass === targetLabel) correctPredictions++;

            let grad: any = dense.backward(targetVectorArray);
            grad = flatten.backward(grad);
            grad = pool.backward(grad);
            grad = relu2d.backward(grad);
            conv.backward(grad);

            layers.forEach(l => l?.updateWeights!(lr));
            dense.update!(lr);
        }

        const avgLoss = (epochLoss / trainImagesFlat.length).toFixed(6);
        const accuracy = ((correctPredictions / trainImagesFlat.length) * 100).toFixed(2);

        console.log(`Epoch ${e + 1} - Avg Loss: ${avgLoss} - Accuracy: ${accuracy}%`);
    }
}

function evaluateMNIST() {
    console.log("--- Starting Evaluation on Test Set ---");
    let correctPredictions = 0;

    const testImagesFlat = mnistData.test.images//.slice(0, 1000);
    const testLabels = mnistData.test.labels//.slice(0, 1000);

    for (let i = 0; i < testImagesFlat.length; i++) {
        const inputMatrix = Matrix.fromArray(testImagesFlat[i], 28, 28);
        const targetLabel = testLabels[i];

        let out: any = conv.forward(inputMatrix);
        out = relu2d.forward(out);
        out = pool.forward(out);
        out = flatten.forward(out);
        const finalOut = dense.forward(out.toArray());

        const probs = finalOut.toArray();
        const predictedClass = probs.indexOf(Math.max(...probs));

        if (predictedClass === targetLabel) correctPredictions++;
    }

    const accuracy = ((correctPredictions / testImagesFlat.length) * 100).toFixed(2);
    console.log(`Test Set Accuracy: ${accuracy}%`);
}

function saveModel() {

    const modelState: ModelState = {
        weights: {
            convWeights: conv.getWeights(),
            denseWeights: dense.getWeights(),
        },
        architecture: [
            {type: "Convo2D", filters: 2, kernelSize: 4, stride: 1, input: {rows: 28, columns: 28}},
            {type: "ReLU2D"},
            {type: "MaxPooling2D", filters: 2, poolSize: 2, stride: 2},
            {type: "Flatten"},
            {
                type: "Dense", layers: [
                    {inputSize: FLATTENED_SIZE, outputSize: 64, activation: ActivationEnum.relu},
                    {inputSize: 64, outputSize: 10, activation: ActivationEnum.softmax}
                ]
            }
        ],
        loss: LossEnum.softmaxce
    };

    fs.writeFileSync('./mnist_model_weights.json', JSON.stringify(modelState, null, 2));
    console.log("Model weights saved successfully to mnist_model_weights.json!");
}

trainMNIST(20, 0.003);

evaluateMNIST();

saveModel();
