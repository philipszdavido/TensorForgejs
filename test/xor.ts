import {ActivationEnum, MSE} from "../src/models";
import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
import {Vector} from "../src/core/Vector";

const inputSize = 2;
const hiddenSize = 4;
const outputSize = 1;

const hiddenLayer = new DenseLayer(inputSize, hiddenSize, ActivationEnum.relu);
const outputLayer = new DenseLayer(hiddenSize, outputSize, ActivationEnum.linear);

const lossFunction = MSE;
const net = new NeuralNetworkDense([hiddenLayer, outputLayer], lossFunction);

const xorDataset = [
    {input: [0, 0], target: [0]},
    {input: [0, 1], target: [1]},
    {input: [1, 0], target: [1]},
    {input: [1, 1], target: [0]}
];

const learningRate = 0.05;
const epochs = 2000;

console.log("Training network on XOR truth table...");
for (let epoch = 0; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (const sample of xorDataset) {
        // Forward
        const pred = net.forward((sample.input));
        totalLoss += lossFunction.loss(Vector.from(sample.target), pred);

        // Backward
        net.backward((sample.target));

        // Gradient Step
        for (const layer of net.denseLayers) {
            (layer as DenseLayer).updateWeights(learningRate);
        }
    }

    if (epoch % 200 === 0) {
        console.log(`Epoch ${epoch} - Loss: ${(totalLoss / 4).toFixed(6)}`);
    }
}

console.log("\n--- Verification ---");
for (const sample of xorDataset) {
    const finalPrediction = net.forward((sample.input));
    const rawValue = finalPrediction.toArray()[0];

    const binaryOutput = rawValue > 0.5 ? 1 : 0;

    console.log(
        `Input: [${sample.input.join(", ")}] ` +
        `-> Target: [${sample.target}] ` +
        `-> Raw Output: ${rawValue.toFixed(4)} ` +
        `-> Classified: ${binaryOutput}`
    );
}
