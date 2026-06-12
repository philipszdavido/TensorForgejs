import {LinearRegression} from "../src/models/LinearRegression";
import LinearRegressionTrainer from "../src/trainers/LinearRegression";
import StochasticGD from "../src/optimizers/SGD";
import Lasso from "../src/regularizers/Lasso";
import MSELoss from "../src/loss/MSELoss";

export const data = [[90], [50], [40], [30], [20], [10]];

const labels = [30, 25, 20, 15, 10, 5];

const linearTrainer = new LinearRegressionTrainer(data, labels, 9);
const linear = new LinearRegression(1)
const regularizer = new Lasso(0.001)
const sgd = new StochasticGD(linear, undefined, 0.0001);

linearTrainer.train(linear, sgd, MSELoss);

console.log(linear.predict([90]));

// (() => {
//     let totalLoss = 0;
//     let epoch = 0;
//
//     for (epoch = 0; epoch < 1000; epoch++) {
//         for (let i = 0; i < bugs.length; i++) {
//
//             // Forward pass
//             const yHat = lReg.forward(bugs[i]);
//
//             lReg.backward(bugs[i], labels[i], yHat);
//
//             const loss = BCELoss(labels[i], yHat);
//
//             totalLoss += loss;
//
//             optimizer.step();
//         }
//     }
//
//     console.log(
//         "Prediction for [4, 1] (Should be close to 1):",
//         lReg.predict([4, 1]),
//     );
//     console.log(
//         "Prediction for [1, 4] (Should be close to 0):",
//         lReg.predict([1, 4]),
//     );
//
//     console.log(`Epoch ${epoch} | Avg Loss: ${totalLoss / bugs.length}`);
// })();
