import {Convo2D} from "../src/models/neural/convo2d/Convo2D";
import {Matrix} from "../src/core/Matrix";
import {MaxPooling2D} from "../src/models/neural/convo2d/MaxPooling2D";

const k = Matrix.random(2, 2)
const cn = new Convo2D(k, 1)

const feature = Matrix.random(5, 5)
k.print("kernel")
feature.print("Feature")
const f = cn.forward(feature)
f.print("Conv")

// console.log(cn.slide_vert_fn(3, 6))
// console.log(cn.slide_down_fn(2, 4))


const max = new MaxPooling2D(Matrix.random(2, 2), 2)
max.forward(f).print("Max Pooling2D")

// const nn = new NeuralNetwork(
//     new Conv2DLayer({ filters: 8, kernelSize: 3 }),
//     new MaxPooling2DLayer({ poolSize: 2, stride: 2 }),
//     new FlattenLayer(),
//
//     new DenseLayer({ inputSize: 1352, outputSize: 128, activation: ActivationEnum.relu }),
//     new DenseLayer({ inputSize: 128,  outputSize: 64,  activation: ActivationEnum.relu }),
//     new DenseLayer({ inputSize: 64,   outputSize: 10,  activation: ActivationEnum.softmax })
// );
