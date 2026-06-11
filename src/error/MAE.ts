import assert from "../assert/assert";
import {Vector} from "../core/Vector";

export default function MAE(x: number[], y: number[]) {
    assert(x.length !== y.length);

    let sum = 0;

    for (let j = 0; j < x.length; j++) {
        sum += Math.abs((sum += x[j] - y[j]));
    }

    return sum / x.length;
}

export function MAEVector(x: Vector, y: Vector) {
    return MAE(x.toArray(), y.toArray())
}

export function MAEGradient(y: Vector, pred: Vector) {
    return Vector.from(pred.toArray().map((p, i) => {
        const act = y.get(i);
        return (p > act ? 1 : p < act ? -1 : 0) / y.length;
    }));
}
