// Binary Cross Entropy
import {Vector} from "../core/Vector";

export default function BCELoss(yHat: number, y: number) {
    const eps = 1e-15;

    yHat = Math.max(eps, Math.min(1 - eps, yHat));

    return -(y * Math.log(yHat) + (1 - y) * Math.log(1 - yHat));
}

export function BCEVector(y: Vector, pred: Vector) {
    return -y.toArray().reduce((s, v, i) => {
        const p = Math.max(pred.get(i), 1e-15);
        return s + v * Math.log(p) + (1 - v) * Math.log(Math.max(1 - p, 1e-15));
    }, 0) / y.length
}

export function BCEGradient(y: Vector, pred: Vector) {
    return Vector.from(pred.toArray().map((p, i) => {
        const t = y.get(i);
        const eps = 1e-15;
        return -(t / Math.max(p, eps) - (1 - t) / Math.max(1 - p, eps)) / y.length;
    }));
}
