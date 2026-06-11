import {Vector} from "../core/Vector";

export default function relu(x: number) {
    return Math.max(0, x)
}

export function ReLU(vec: Vector) {

    const v = new Vector(vec.length);

    for (let i = 0; i < vec.length; i++) {
        v.set(i, relu(vec.get(i)));
    }

    return v;

}

export function ReLU_derivative(v: Vector, _: any): Vector {
    const result = new Vector(v.length);

    for (let i = 0; i < v.length; i++) {
        result.set(i, v.get(i) > 0 ? 1 : 0);
    }

    return result;
}
