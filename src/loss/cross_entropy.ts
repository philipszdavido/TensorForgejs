import {Vector} from "../core/Vector";

export function CrossEntropyLoss(y: Vector, pred: Vector) {
    return -y.toArray().reduce((s, v, i) => s + v * Math.log(Math.max(pred.get(i), 1e-15)), 0)
}

export function CrossEntropyGradient(y: Vector, pred: Vector) {
    return Vector.from(pred.toArray().map((p, i) => -y.get(i) / Math.max(p, 1e-15)))
}
