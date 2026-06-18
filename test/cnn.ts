// import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
// import {Matrix} from "../src/core/Matrix";
// import {Flatten} from "../src/models/neural/convo2d/Flatten";
// import {ReLU2D} from "../src/models/neural/convo2d/ReLU2DLayer";
// import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
// import DenseLayer from "../src/models/neural/dense/DenseLayer";
// import {ActivationEnum, SoftmaxCE, SoftmaxCrossEntropy} from "../src/models";
//
// const conv =
//     new Convo2D(
//         Matrix.random(3, 3),
//         1
//     );
//
// const relu2d =
//     new ReLU2D();
//
// const flatten =
//     new Flatten();
//
// const dense =
//     new NeuralNetworkDense(
//         [
//             new DenseLayer(
//                 26 * 26,
//                 128,
//                 ActivationEnum.relu
//             ),
//
//             new DenseLayer(
//                 128,
//                 10,
//                 ActivationEnum.softmax
//             )
//         ],
//         SoftmaxCE
//     );
//
// const f = conv.forward(Matrix.toMatrix(new Array(28 * 28).fill(Math.random())))
// f.print("Conv")
//
// const r = relu2d.forward(f)
// r.print("RelU")
//
// const flat = flatten.forward(r)
//
// // flat.print("Flatten")
//
// const d = dense.forward(flat.toArray())
// d.print("DenseLayer")
//
// const dFlat = dense.backward()

// ===============

// import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
// import {Matrix} from "../src/core/Matrix";
// import {Flatten} from "../src/models/neural/convo2d/Flatten";
// import {ReLU2D} from "../src/models/neural/convo2d/ReLU2DLayer";
// import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
// import DenseLayer from "../src/models/neural/dense/DenseLayer";
// import {ActivationEnum, SoftmaxCE} from "../src/models";
// import {MaxPooling2D} from "../src/models/neural/convo2d/MaxPooling2D";
//
// const kernelWeights = Matrix.toMatrix([
//     1, 0, -1,
//     0, 1, 0,
//     -1, 0, 1
// ], 3, 3);
//
// kernelWeights.print("Kernel Weights");
//
// const conv = new Convo2D(kernelWeights, 1);
// const relu2d = new ReLU2D();
// const pool = new MaxPooling2D(Matrix.fromArray([1], 1, 1), 1);
// const flatten = new Flatten();
//
// const dense = new NeuralNetworkDense(
//     [
//         new DenseLayer(2 * 2, 5, ActivationEnum.relu),
//         new DenseLayer(5, 3, ActivationEnum.softmax)
//     ],
//     SoftmaxCE
// );
//
// const inputData = Matrix.fromArray([
//     1, 2, 3, 0,
//     0, 1, 2, 1,
//     2, 0, 1, 2,
//     3, 1, 0, 1
// ], 4, 4);
//
// const targetData = [0.0, 1.0, 0.0];
//
// console.log("--- Running Forward Pass ---");
//
// const f = conv.forward(inputData);
// f.print("Conv Output (Expected: [[-2, 1], [2, -1]])");
//
// const r = relu2d.forward(f);
// r.print("ReLU Output (Expected: [[0, 1], [2, 0]])");
//
// const p = pool.forward(r);
// p.print("Max Pool")
//
// const flat = flatten.forward(p);
// console.log("Flattened Array:", flat.toArray(), "(Expected: [0, 1, 2, 0])");
//
// const d = dense.forward(flat.toArray());
// d.print("Dense Softmax Output (Should sum to 1.0)");
//
// console.log("\n--- Running Backward Pass ---");
//
// const denseGradFlat = dense.backward(targetData);
// console.log("Gradient exiting Dense layer (Length should be 4):", denseGradFlat);
//
// // const denseGradMatrix = Matrix.fromArray(denseGradFlat, 2, 2);
//
// const flattenGrad = flatten.backward(denseGradFlat);
// flattenGrad.print("Flatten Grad");
//
// const poolGrad = pool.backward(flattenGrad);
// poolGrad.print("Max Pool Grad");
//
// const reluGrad = relu2d.backward(poolGrad);
// reluGrad.print("RELU Grad")
// const convGrad = conv.backward(reluGrad);
//
// convGrad.print("ConvGrad")
//
// console.log("\nBackward pass completed cleanly!");
//
// console.log("\n--- Testing Model Predictions ---");
//
// const result = predict(inputData);
//
// console.log(`Predicted Class Index: ${result.classIndex}`);
// console.log(`Confidence Scores:`, result.probabilities);
//
// if (result.classIndex === 1) {
//     console.log("✅ Success! The model correctly detected Class 1.");
// } else {
//     console.log("❌ Incorrect prediction. The model might need more training epochs.");
// }
//
// const unseenInput = Matrix.fromArray([
//     0, 1, 3, 1,
//     1, 1, 2, 0,
//     1, 0, 2, 2,
//     2, 1, 0, 0
// ], 4, 4);
//
// const unseenResult = predict(unseenInput);
// console.log(`\nNew Image Prediction: Class ${unseenResult.classIndex}`);
//
// function predict(inputImage: Matrix): { classIndex: number; probabilities: number[] } {
//
//     const f = conv.forward(inputImage);
//     const r = relu2d.forward(f);
//     const flat = flatten.forward(r);
//
//     const denseOutput = dense.forward(flat.toArray());
//     const probabilities = denseOutput.toArray();
//
//     let predictedClass = 0;
//     let maxProb = -Infinity;
//
//     for (let i = 0; i < probabilities.length; i++) {
//         if (probabilities[i] > maxProb) {
//             maxProb = probabilities[i];
//             predictedClass = i;
//         }
//     }
//
//     return {
//         classIndex: predictedClass,
//         probabilities: probabilities
//     };
// }

// ========

import {Matrix} from "../src/core/Matrix";
import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
import {ReLU2D} from "../src/models/neural/convo2d/ReLU2DLayer";
import {MaxPooling2D} from "../src/models/neural/convo2d/MaxPooling2D";
import {Flatten} from "../src/models/neural/convo2d/Flatten";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {ActivationEnum, LayerInterface, SoftmaxCE} from "../src/models";
import {Vector} from "../src/core/Vector";

const conv = new Convo2D([Matrix.random(3, 3)], 1);
const relu2d = new ReLU2D();
const pool = new MaxPooling2D([Matrix.fromArray([1, 1, 1, 1], 2, 2)], 2);
const flatten = new Flatten();
const lossFunction = SoftmaxCE
const dense = new NeuralNetworkDense([
    new DenseLayer(1, 5, ActivationEnum.relu),
    new DenseLayer(5, 3, ActivationEnum.softmax)
], lossFunction);

const layers: LayerInterface[] = [conv, relu2d, pool, flatten];

let totalLoss = 0;

function _train(input: Matrix, target: number[], epochs: number, lr: number) {
    console.log(`--- Starting Training (${epochs} epochs) ---`);

    for (let e = 0; e < epochs; e++) {
        // Forward
        let out: Matrix | Matrix[] | Vector = conv.forward(input);
        out = relu2d.forward(out);
        out = pool.forward(out);
        out = flatten.forward(out);
        const finalOut = dense.forward(out.toArray());

        // finalOut.print("Predicted")
        // Vector.from(target).print("Target")

        totalLoss += lossFunction.loss(Vector.from(target), finalOut);

        let grad: Matrix | Matrix[] | Vector = dense.backward(target);
        grad = flatten.backward(grad);
        grad = pool.backward(grad);
        grad = relu2d.backward(grad);
        conv.backward(grad);

        layers.forEach(l => l?.updateWeights!(lr));
        dense.update!(lr);

        if (e % 200 === 0) {
            console.log(`Epoch ${e} - Loss: ${(totalLoss / 4).toFixed(6)}`);
        }

    }
    console.log("--- Training Complete ---");
}

function train(input: Matrix, target: number[], epochs: number, lr: number) {
    console.log(`--- Starting Training (${epochs} epochs) ---`);

    for (let e = 0; e < epochs; e++) {

        let out: any = conv.forward(input);
        out = relu2d.forward(out);
        out = pool.forward(out);
        out = flatten.forward(out);
        const finalOut = dense.forward(out.toArray());

        const currentLoss = lossFunction.loss(Vector.from(target), finalOut);

        let grad: any = dense.backward(target);
        grad = flatten.backward(grad);
        grad = pool.backward(grad);
        grad = relu2d.backward(grad);
        conv.backward(grad);

        layers.forEach(l => l?.updateWeights!(lr));
        dense.update!(lr);

        if (e % 200 === 0) {
            console.log(`Epoch ${e} - Loss: ${currentLoss.toFixed(6)}`);
        }
    }
}

function predict(input: Matrix) {
    let out: Matrix | Matrix[] | Vector = conv.forward(input);
    out = relu2d.forward(out);
    out = pool.forward(out);
    out = flatten.forward(out);
    const probs = dense.forward(out.toArray()).toArray();

    const classIndex = probs.indexOf(Math.max(...probs));
    return {classIndex, probabilities: probs};
}

const inputData = Matrix.fromArray([1, 2, 3, 0, 0, 1, 2, 1, 2, 0, 1, 2, 3, 1, 0, 1], 4, 4);
const targetData = [0.0, 1.0, 0.0];

train(inputData, targetData, 5000, 0.01);

const result = predict(inputData);
console.log("Final Prediction:", result);

function trainTestSplit(data: DatasetEntry[], testRatio: number = 0.2) {

    const shuffled = [...data].sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * (1 - testRatio));

    return {
        trainSet: shuffled.slice(0, splitIndex),
        testSet: shuffled.slice(splitIndex)
    };
}

type DatasetEntry = {
    input: Matrix;
    target: number[];
};

const dataset: DatasetEntry[] = [
    { input: Matrix.fromArray([1, 2, 3, 0...], 4, 4), target: [0, 1, 0] },
    { input: Matrix.fromArray([0, 1, 0, 1...], 4, 4), target: [1, 0, 0] },
];

const { trainSet, testSet } = trainTestSplit(dataset);

function runTraining(epochs: number, lr: number) {
    for (let e = 0; e < epochs; e++) {
        let epochLoss = 0;
        for (const { input, target } of trainSet) {
            epochLoss += currentLoss;
        }
        if (e % 100 === 0) console.log(`Epoch ${e} - Avg Loss: ${epochLoss / trainSet.length}`);
    }
}

function evaluate(testSet: DatasetEntry[]) {
    let correct = 0;
    for (const { input, target } of testSet) {
        const prediction = predict(input);
        const targetIndex = target.indexOf(Math.max(...target));
        if (prediction.classIndex === targetIndex) correct++;
    }
    console.log(`Accuracy: ${(correct / testSet.length) * 100}%`);
}
