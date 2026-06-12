import PerceptronTrainer from "../src/trainers/PerceptronTrainer";
import DiscretePerceptron from "../src/models/Perceptron";

const bugs = [
    [1, 3],
    [3, 1],
];

const labels = [0, 1];

const trainer = new PerceptronTrainer(bugs, labels);
const model = new DiscretePerceptron(2)

trainer.train(model);

console.log(
    "Prediction for [4, 1] (Should be close to 1):",
    model.forward([4, 1]),
);
console.log(
    "Prediction for [1, 4] (Should be close to 0):",
    model.forward([1, 4]),
);
