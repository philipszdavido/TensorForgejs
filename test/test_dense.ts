import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {ActivationEnum, MSE} from "../src/models";
import {Vector} from "../src/core/Vector";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";

const inputSize = 3;
const hiddenSize = 4;
const outputSize = 2;

const hiddenLayer = new DenseLayer(inputSize, hiddenSize, ActivationEnum.relu);
const outputLayer = new DenseLayer(hiddenSize, outputSize, ActivationEnum.linear);
const lossFunction = MSE

const net = new NeuralNetworkDense(
    [hiddenLayer, outputLayer],
    lossFunction
);

export const XOR_DATASET = [
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [0]}
];

const trainingData = [
    {input: [1.0, 2.0, -0.5], target: [0.0, 1.0]},
    {input: [0.5, -1.0, 2.0], target: [1.0, 0.0]}
];

const learningRate = 0.01;
const epochs = 1000;

for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (const sample of trainingData) {

        const prediction = net.forward(sample.input);

        totalLoss += lossFunction.loss(Vector.from(sample.target), prediction);

        net.backward(sample.target);

        for (const layer of net.denseLayers) {
            //(layer as DenseLayer).updateWeights(learningRate);
        }
        net.update(learningRate);
    }

    if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch} - Loss: ${totalLoss / trainingData.length}`);
    }
}

console.log("Training complete! Testing inference...");
const unseenInput = [1.0, 2.0, -0.5];
const finalPrediction = net.forward(unseenInput);

console.log("Prediction:", finalPrediction.toArray());
