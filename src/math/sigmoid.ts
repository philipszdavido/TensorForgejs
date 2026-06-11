import {Vector} from "../core/Vector";

export default function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

export function Sigmoid(vec: Vector) {

    const v = new Vector(vec.length);

    for (let i = 0; i < vec.length; i++) {
        v.set(i, sigmoid(vec.get(i)));
    }

    return v;

}

export function Sigmoid_derivative(_: any, a: Vector): Vector {
    const v = new Vector(a.length);

    for (let i = 0; i < a.length; i++) {
        const val = a.get(i);
        v.set(i, val * (1 - val));
    }

    return v;
}
