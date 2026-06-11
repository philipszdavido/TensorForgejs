import {Vector} from "../../core/Vector";
import {Matrix} from "../../core/Matrix";

export type Activation = {
    forward(x: Vector): Vector;
    derivative(x: Vector, y: Vector): Vector;
    initializer(inputs: number): number;
};

export type LossFunction = {
    loss(y: Vector, pred: Vector): number;
    gradient(y: Vector, pred: Vector): Vector;
};

export type Input = {
    size: number;
}

export type Hidden = {
    size: number;
    activation: Activation;
}

export type Output = {
    size: number;
    activation: Activation;
};

export type Layer = {
    weight: Matrix, // W (out × in)
    bias: Vector, // b (out)
    dW: Matrix,
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation

    activation: Activation;
};
