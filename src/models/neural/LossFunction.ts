import {LossFunction} from "./types";
import {meanSquareErrorVector, MSEGradient} from "../../error/mse";
import {BCEGradient, BCEVector} from "../../loss/BCELoss";

export const MSE: LossFunction = {
    loss: meanSquareErrorVector,
    gradient: MSEGradient
};

export const BCE: LossFunction = {
    loss: BCEVector,
    gradient: BCEGradient
};
