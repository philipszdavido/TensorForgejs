import {Vector} from "../../core/Vector";
import {Activation} from "./types";
import {ReLU, ReLU_derivative} from "../../math/relu";
import {Sigmoid, Sigmoid_derivative} from "../../math/sigmoid";
import {SoftmaxCrossEntropy} from "./SoftmaxCrossEntropy";

export const LinearActivation: Activation = {
    forward: (x) => x,
    derivative: (x) => {
        const data = new Array(x.length).fill(1);
        return Vector.from(data);
    },
    initializer(inputs) {
        return Math.sqrt(1 / inputs);
    }
};

export const ReLUActivation: Activation = {
    forward: ReLU,
    derivative: ReLU_derivative,
    initializer(inputs) {
        return Math.sqrt(2 / inputs);
    }
};

export const SigmoidActivation: Activation = {
    forward: Sigmoid,
    derivative: Sigmoid_derivative,
    initializer(inputs) {
        return Math.sqrt(1 / inputs);
    }
};

export const SoftmaxCE = new SoftmaxCrossEntropy();
export const SoftmaxPassThrough = {
    forward: (x: any) => SoftmaxCE.forward(x),
    derivative: () => {
        throw new Error("Fused");
    },
    initializer: (inputs: number) => Math.sqrt(1 / inputs)
};
