// import {Tensor} from "../../core/Tensor";
//
// type Activation = "relu" | "sigmoid" | "tanh" | "linear";
//
// function activate(x: number, type: Activation): number {
//     switch (type) {
//         case "relu":
//             return Math.max(0, x);
//         case "sigmoid":
//             return 1 / (1 + Math.exp(-x));
//         case "tanh":
//             return Math.tanh(x);
//         case "linear":
//             return x;
//     }
// }
//
// function activateDerivative(x: number, type: Activation): number {
//     switch (type) {
//         case "relu":
//             return x > 0 ? 1 : 0;
//         case "sigmoid":
//             const s = 1 / (1 + Math.exp(-x));
//             return s * (1 - s);
//         case "tanh":
//             const t = Math.tanh(x);
//             return 1 - t * t;
//         case "linear":
//             return 1;
//     }
// }
//
// interface Layer {
//     W: Tensor;
//     b: Tensor;
//     activation: Activation;
//
//     z?: Tensor; // pre-activation
//     a?: Tensor; // activation output
//     input?: Tensor;
// }
//
// export class NeuralNetwork {
//     private layers: Layer[] = [];
//     private learningRate: number;
//
//     constructor(learningRate = 0.01) {
//         this.learningRate = learningRate;
//     }
//
//     add(inputSize: number, outputSize: number, activation: Activation = "relu") {
//         const W = Tensor.from2DArray(
//             Array.from({length: outputSize}, () =>
//                 Array.from({length: inputSize}, () => (Math.random() - 0.5) * 0.1)
//             )
//         );
//
//         const b = Tensor.fromArray(
//             Array.from({length: outputSize}, () => 0)
//         );
//
//         this.layers.push({W, b, activation});
//     }
//
//     forward(input: Tensor): Tensor {
//         let a = input;
//
//         for (const layer of this.layers) {
//             layer.input = a;
//
//             const z = layer.W.matMul(a).add(layer.b);
//             layer.z = z;
//
//             const activated = Tensor.fromArray(
//                 z.toArray().map(v => activate(v, layer.activation))
//             );
//
//             layer.a = activated;
//             a = activated;
//         }
//
//         return a;
//     }
//
//     private mse(pred: Tensor, target: Tensor): number {
//         const p = pred.toArray();
//         const t = target.toArray();
//
//         let sum = 0;
//         for (let i = 0; i < p.length; i++) {
//             const d = p[i] - t[i];
//             sum += d * d;
//         }
//
//         return sum / p.length;
//     }
//
//     backward(pred: Tensor, target: Tensor) {
//         const outputLayer = this.layers[this.layers.length - 1];
//
//         let grad = pred.toArray().map((p, i) => 2 * (p - target.toArray()[i]));
//
//         for (let l = this.layers.length - 1; l >= 0; l--) {
//             const layer = this.layers[l];
//             const input = layer.input!;
//             const z = layer.z!;
//
//             const dz = new Array(grad.length);
//
//             for (let i = 0; i < grad.length; i++) {
//                 dz[i] = grad[i] * activateDerivative(z.get(i), layer.activation);
//             }
//
//             const inputArr = input.toArray();
//
//             const dW = Tensor.from2DArray(
//                 dz.map(d =>
//                     inputArr.map(x => d * x)
//                 )
//             );
//
//             const db = Tensor.fromArray(dz);
//
//             layer.W = layer.W.sub(dW.mulScalar(this.learningRate));
//             layer.b = layer.b.sub(db.mulScalar(this.learningRate));
//
//             const W_T = this.transpose(layer.W);
//             const newGrad = W_T.matMul(Tensor.fromArray(dz)).toArray();
//
//             grad = newGrad;
//         }
//     }
//
//     trainStep(input: Tensor, target: Tensor): number {
//         const pred = this.forward(input);
//         const loss = this.mse(pred, target);
//         this.backward(pred, target);
//         return loss;
//     }
//
//     private transpose(t: Tensor): Tensor {
//         const [rows, cols] = t.shape;
//
//         const result = new Tensor([cols, rows]);
//
//         for (let i = 0; i < rows; i++) {
//             for (let j = 0; j < cols; j++) {
//                 result.set(t.get(i, j), j, i);
//             }
//         }
//
//         return result;
//     }
// }
//
// const nn = new NeuralNetwork(0.01);
//
// nn.add(2, 4, "relu");
// nn.add(4, 1, "sigmoid");
//
// const x = Tensor.fromArray([1, 0]);
// const y = Tensor.fromArray([1]);
//
// for (let i = 0; i < 1000; i++) {
//     const loss = nn.trainStep(x, y);
//     if (i % 100 === 0) console.log(loss);
// }
// nn.forward(x).prettyPrint("Output")
//
// export const xorData = [
//     {
//         input: Tensor.fromArray([0, 0]),
//         target: Tensor.fromArray([0])
//     },
//     {
//         input: Tensor.fromArray([0, 1]),
//         target: Tensor.fromArray([1])
//     },
//     {
//         input: Tensor.fromArray([1, 0]),
//         target: Tensor.fromArray([1])
//     },
//     {
//         input: Tensor.fromArray([1, 1]),
//         target: Tensor.fromArray([0])
//     }
// ];
//
// const nn = new NeuralNetwork(0.1);
//
// nn.add(2, 4, "tanh");
// nn.add(4, 1, "sigmoid");
//
// for (let i = 0; i < 10000; i++) {
//     const sample = xorData[Math.floor(Math.random() * xorData.length)];
//     nn.trainStep(sample.input, sample.target);
// }
//
// nn.forward(xorData[0].input).prettyPrint()
// nn.forward(xorData[1].input).prettyPrint()
// nn.forward(xorData[2].input).prettyPrint()
// nn.forward(xorData[3].input).prettyPrint()

import {LossFunction, NLayer} from "../Types";
import {Tensor} from "../../../core/Tensor";
import {Vector} from "../../../core/Vector";
import {SoftmaxCrossEntropy} from "../SoftmaxCrossEntropy";

export class NeuralNetwork {

    constructor(
        private readonly layers: NLayer[],
        public readonly loss: LossFunction,
    ) {
    }

    add(layer: NLayer) {
        this.layers.push(layer);
    }

    forward(input: Tensor): Tensor {
        let x = input;

        for (const layer of this.layers) {
            x = layer.forward(x);
        }

        return x;
    }

    backward(pred: Tensor, target: Tensor) {

        // this is the output, we are using MSE gradient to calc grad
        let grad = pred.toArray().map((p, i) =>
            2 * (p - target.toArray()[i])
        );

        let g = Tensor.fromArray(grad);

        for (let i = this.layers.length - 1; i >= 0; i--) {
            g = this.layers[i].backward(g);
        }
    }

    trainStep(input: Tensor, target: Tensor): number {
        const pred = this.forward(input);

        const loss =
            pred.toArray().reduce((s, p, i) =>
                    s + Math.pow(p - target.toArray()[i], 2)
                , 0) / pred.toArray().length;

        this.backward(pred, target);

        return loss;
    }

    updateWeights(): void {
        for (const layer of this.layers) {
            layer.updateWeights();
        }
    }

}
