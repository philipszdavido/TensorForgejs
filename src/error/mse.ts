import assert from "../assert/assert";
import {Vector} from "../core/Vector";

// Mean Square Error
export default function meanSquareError(x: number[], y: number[]) {

    assert(x.length == y.length);

    let sum = 0
    for (let i = 0; i < x.length; i++) {
        const result = (x[i] - y[i]) ** 2
        sum += result
    }

    return sum / x.length;
}

export function meanSquareErrorVector(y: Vector, pred: Vector) {
    return meanSquareError(y.toArray(), pred.toArray())
}

export function MSEGradient(y: Vector, pred: Vector) {
    return Vector.from(pred.toArray().map((p, i) => (p - y.get(i)) / y.length))
}
