import {LossFunction} from "./Types";
import {meanSquareErrorVector, MSEGradient} from "../../error/mse";
import {BCEGradient, BCEVector} from "../../loss/BCELoss";
import {SoftmaxCE} from "./Activation";

export const MSE: LossFunction = {
    loss: meanSquareErrorVector,
    gradient: MSEGradient
};

export const BCE: LossFunction = {
    loss: BCEVector,
    gradient: BCEGradient
};

export enum LossEnum {
    mce,
    bce,
    softmaxce
}

export const Loss = {
    [LossEnum.mce]: MSE,
    [LossEnum.bce]: BCE,
    [LossEnum.softmaxce]: SoftmaxCE,
}
