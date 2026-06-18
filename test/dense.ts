import {NeuralNetwork} from "../src/models/neural/Tensor/NeuralNetworkTensor";
import {DenseLayer} from "../src/models/neural/Tensor/DenseLayer";
import {Tensor} from "../src/core/Tensor";

export const xorData = [
    {
        input: Tensor.fromArray([0, 0]),
        target: Tensor.fromArray([0])
    },
    {
        input: Tensor.fromArray([0, 1]),
        target: Tensor.fromArray([1])
    },
    {
        input: Tensor.fromArray([1, 0]),
        target: Tensor.fromArray([1])
    },
    {
        input: Tensor.fromArray([1, 1]),
        target: Tensor.fromArray([0])
    }
];

const nn = new NeuralNetwork();

nn.add(new DenseLayer(2, 4, "tanh", 0.1));
nn.add(new DenseLayer(4, 1, "sigmoid", 0.1));

for (let i = 0; i < 10000; i++) {
    const sample = xorData[Math.floor(Math.random() * xorData.length)];
    nn.trainStep(sample.input, sample.target);
}

nn.forward(xorData[0].input).prettyPrint()
nn.forward(xorData[1].input).prettyPrint()
nn.forward(xorData[2].input).prettyPrint()
nn.forward(xorData[3].input).prettyPrint()
