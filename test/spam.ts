import {NeuralNetwork, BCE, ReLUActivation, SigmoidActivation} from "../src/models/neural";
import {writeFileSync} from "node:fs";

const vocabulary = ["buy", "now", "free", "meeting", "project", "click", "hello"];
const vocabSize = vocabulary.length;

function exportNetworkToJSON(network: NeuralNetwork, vocabulary: string[]): string {
    const modelData = {
        weights: network.layers.map(layer => layer.weight.toNestedArray()),
        biases: network.layers.map(layer => layer.bias.toArray()),

        architecture: {
            inputSize: network.input.size,
            hiddenLayers: network.hidden.map(h => h.size),
            outputSize: network.output.size
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
        activation: hiddenActivations[idx] || hiddenActivations[0] // fallback if array or single object passed
    }));

    const outputConfig = {
        size: architecture.outputSize,
        activation: outputActivation
    };

    // 2. Instantiate a clean network shell with zeroed dimensions
    const network = new NeuralNetwork(inputConfig, hiddenConfig, outputConfig, lossFunction);

    // 3. Hydrate the blank matrices and vectors with your trained parameters
    for (let i = 0; i < network.layers.length; i++) {
        const layer = network.layers[i];

        // Rebuild your internal Matrix from the 2D nested array
        layer.weight = Matrix.from(weights[i]);

        // Rebuild your internal Vector from the 1D plain array
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
    [{size: 4, activation: ReLUActivation}],
    {size: 1, activation: SigmoidActivation},
    BCE
);

const trainingData = [
    {text: "buy free tokens now click", isSpam: [1]},
    {text: "hello meeting project presentation", isSpam: [0]}
];

const learningRate = 0.1;

for (let epoch = 0; epoch < 100; epoch++) {
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
console.log(finalJson);
writeFileSync("./spam-model.json", JSON.stringify(finalJson));
