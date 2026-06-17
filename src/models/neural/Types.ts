import {Vector} from "../../core/Vector";
import {Matrix} from "../../core/Matrix";
import {
    LeakyReLUActivation,
    LinearActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxPassThrough, TanhActivation
} from "./Activation";
import {Tensor} from "../../core/Tensor";

export enum ActivationEnum {
    relu,
    softmax,
    sigmoid,
    linear,
    leakyRelu,
    tanh,
}

export const ActivationUse = {
    [ActivationEnum.relu]: ReLUActivation,
    [ActivationEnum.softmax]: SoftmaxPassThrough,
    [ActivationEnum.sigmoid]: SigmoidActivation,
    [ActivationEnum.linear]: LinearActivation,
    [ActivationEnum.leakyRelu]: LeakyReLUActivation,
    [ActivationEnum.tanh]: TanhActivation,
};

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
    activation: ActivationEnum;
}

export type Output = {
    size: number;
    activation: ActivationEnum;
};

export type Layer = {
    weight: Matrix, // W (out × in)
    bias: Vector, // b (out)
    dW: Matrix,
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation

    activation: ActivationEnum;
};

export interface NLayer {
    forward: (input: Tensor) => Tensor;
    backward: (input: Tensor) => Tensor;

    updateWeights(): void;

    parameters(): Tensor[]

    gradients(): Tensor[]
}

export interface LayerInterface {

    forward(input: any): any;

    backward(grad: any): any;

    updateWeights?(lr: number): void;

}
