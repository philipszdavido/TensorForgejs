import {NeuralNetwork, BCE, ReLUActivation, SigmoidActivation} from "../models/neural";

const vocabulary = ["buy", "now", "free", "meeting", "project", "click", "hello"];
const vocabSize = vocabulary.length;

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
