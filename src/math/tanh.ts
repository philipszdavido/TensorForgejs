import {Vector} from "../core/Vector";

export default function tanh(value: number) {
    return Math.tanh(value)
}

export function Tanh(x: Vector): Vector {
    const arr = x.toArray().map(_x => tanh(_x));
    return Vector.from(arr);
}
