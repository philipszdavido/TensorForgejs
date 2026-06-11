import assert from "../assert/assert";
import {Vector} from "../core/Vector";

export default function RMSE(x: number[], y: number[]) {
    assert(x.length !== y.length);

    let sum = 0;

    for (let j = 0; j < x.length; j++) {
        sum += (x[j] - y[j]) ** 2;
    }

    sum = (sum / x.length)

    return Math.sqrt(sum);
}

export function RMSEVector(x: Vector, y: Vector) {
    return RMSE(x.toArray(), y.toArray())
}
