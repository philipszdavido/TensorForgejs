import {Vector} from "../../core/Vector";
import {Activation} from "./Types";
import {ReLU, ReLU_derivative} from "../../math/relu";
import {Sigmoid, Sigmoid_derivative} from "../../math/sigmoid";
import {SoftmaxCrossEntropy} from "./SoftmaxCrossEntropy";
import He from "../../math/initialization/He";
import Xavier from "../../math/initialization/Xavier";
import {Tanh} from "../../math/tanh";

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

export const LeakyReLUActivation: Activation = {
    derivative(x: Vector, y: Vector): Vector {
        const arr = y.toArray().map(_y => _y > 0 ? 1 : 0.01);
        return Vector.from(arr);
    }, forward(x: Vector): Vector {
        const arr = x.toArray().map(_x => _x > 0 ? _x : 0.01 * _x);
        return Vector.from(arr);
    }, initializer: He

}

export const TanhActivation: Activation = {
    derivative(x: Vector, y: Vector): Vector {
        const arr = x.toArray().map(actVal => 1 - (actVal ** 2));
        return Vector.from(arr);
    }, forward: Tanh, initializer: Xavier

}

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
