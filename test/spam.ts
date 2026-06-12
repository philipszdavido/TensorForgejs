import {ActivationEnum, BCE, LossFunction, NeuralNetwork} from "../src/models/neural";
import {writeFileSync} from "node:fs";
import {Matrix} from "../src/core/Matrix";
import {Vector} from "../src/core/Vector";

const vocabulary = ["buy", "now", "free", "meeting", "project", "click", "hello"];
const vocabSize = vocabulary.length;

function exportNetworkToJSON(network: NeuralNetwork, vocabulary: string[]): string {
    const modelData = {
        weights: network.layers.map(layer => layer.weight.toNestedArray()),
        biases: network.layers.map(layer => layer.bias.toArray()),

        architecture: {
            input: network.input,
            hiddenLayers: network.hidden,
            output: network.output
        },

        vocabulary
    };

    return JSON.stringify(modelData, null, 2);
}

function fromJSON(jsonString: string, lossFunction: LossFunction, hiddenActivations: any[], outputActivation: any): NeuralNetwork {
    const modelData = JSON.parse(jsonString);
    const {architecture, weights, biases} = modelData;

    const inputConfig = {size: architecture.inputSize};

    const hiddenConfig = architecture.hiddenLayers.map((size: number, idx: number) => ({
        size,
        activation: hiddenActivations[idx] || hiddenActivations[0]
    }));

    const outputConfig = {
        size: architecture.outputSize,
        activation: outputActivation
    };

    // Instantiate a clean network shell with zeroed dimensions
    const network = new NeuralNetwork(inputConfig, hiddenConfig, outputConfig, lossFunction);

    // Hydrate the blank matrices and vectors with your trained parameters
    for (let i = 0; i < network.layers.length; i++) {
        const layer = network.layers[i];

        layer.weight = Matrix.from(weights[i]);

        layer.bias = Vector.from(biases[i]);
    }

    return network;
}

function textToVector(text: string): number[] {
    const tokens = text.toLowerCase().split(/\s+/);
    const vector = new Array(vocabSize).fill(0);
    tokens.forEach(token => {
        const idx = vocabulary.indexOf(token);
        if (idx !== -1) vector[idx] = 1;
    });
    return vector;
}

const spamNet = new NeuralNetwork(
    {size: vocabSize},
    [{size: 4, activation: ActivationEnum.relu}],
    {size: 1, activation: ActivationEnum.sigmoid},
    BCE
);

const trainingData = [
    {text: "buy free tokens now click", isSpam: [1]},
    {text: "hello meeting project presentation", isSpam: [0]}
];

const learningRate = 0.1;

for (let epoch = 0; epoch < 1000; epoch++) {
    for (const item of trainingData) {
        const inputVector = textToVector(item.text);

        spamNet.forward(inputVector);

        spamNet.backward(item.isSpam);

        spamNet.update(learningRate);
    }
}

const testVector = textToVector("click here to buy free stuff");
const prediction = spamNet.forward(testVector);
console.log(`Spam Probability: ${prediction.get(0)}`);

const finalJson = exportNetworkToJSON(spamNet, vocabulary);

writeFileSync("./spam-model.json", (finalJson));
