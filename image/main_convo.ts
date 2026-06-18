import {loadLocalMNIST} from "./mnistLoader";
import {NeuralNetwork, SoftmaxCE, ActivationEnum} from "../src/models/neural";
import {Vector} from "../src/core/Vector";
import {writeFileSync} from "node:fs";
import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
import {Matrix} from "../src/core/Matrix";
import {ReLU2D} from "../src/models/neural/convo2d/ReLU2DLayer";
import {Flatten} from "../src/models/neural/convo2d/Flatten";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {CNNNetwork} from "../src/models/neural/cnn/CNNNetwork";

function oneHot(digit: number): number[] {
    const arr = new Array(10).fill(0);
    arr[digit] = 1;
    return arr;
}

function exportNetworkToJSON(network: NeuralNetwork): string {
    const modelData = {
        weights: network.layers.map(layer => layer.weight.toNestedArray()),
        biases: network.layers.map(layer => layer.bias.toArray()),

        architecture: {
            input: network.input,
            hiddenLayers: network.hidden.map(h => ({size: h.size, activation: h.activation})),
            output: network.output
        }
    };

    return JSON.stringify(modelData, null, 2);
}

function runPipeline() {

    console.log("Loading dataset from local data/mnist/ files...");
    const dataset = loadLocalMNIST();

    const nn = CNN()

    const epochs = 10;
    const learningRate = 0.02;
    const totalTrainingSamples = dataset.train.images.length;

    console.log("\nStarting training passes...");
    // nn.setMode('train');

    for (let epoch = 1; epoch <= epochs; epoch++) {
        let epochLoss = 0;

        for (let i = 0; i < totalTrainingSamples; i++) {
            const x = dataset.train.images[i];
            const y = oneHot(dataset.train.labels[i]);

            const pred = nn.forward(Matrix.toMatrix(x));
            nn.backward(y);
            nn.update(learningRate);

            epochLoss += SoftmaxCE.loss(Vector.from(y), pred);
        }

        console.log(`Epoch ${epoch}/${epochs} | Avg Loss: ${(epochLoss / totalTrainingSamples).toFixed(4)}`);
    }

    console.log("\nVerifying model performance against evaluation set...");
    // nn.setMode('eval');
    let successfulHits = 0;

    for (let i = 0; i < dataset.test.images.length; i++) {
        const outVec = nn.forward(Matrix.toMatrix(dataset.test.images[i]));
        const arr = outVec.toArray();

        const prediction = arr.indexOf(Math.max(...arr));
        const actual = dataset.test.labels[i];

        if (prediction === actual) successfulHits++;
    }

    const accuracy = (successfulHits / dataset.test.images.length) * 100;
    console.log(`Evaluation Complete. Accuracy: ${accuracy.toFixed(2)}%\n`);

    // const finalJson = exportNetworkToJSON(nn);

    // writeFileSync("./image28x28-model.json", (finalJson));

}

function CNN() {

    const conv =
        new Convo2D(
            Matrix.random(3, 3),
            1
        );

    const relu =
        new ReLU2D();

    const flatten =
        new Flatten();

    const dense =
        new NeuralNetworkDense(
            [
                new DenseLayer(
                    26 * 26,
                    128,
                    ActivationEnum.relu
                ),

                new DenseLayer(
                    128,
                    10,
                    ActivationEnum.softmax
                )
            ],
            SoftmaxCE
        );

    return new CNNNetwork(
        conv,
        relu,
        flatten,
        dense
    );
}

runPipeline();
