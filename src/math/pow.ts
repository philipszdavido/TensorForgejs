import {Vector} from "../core/Vector";

export function Pow(x: Vector, value: number) {
    for (let i = 0; i < x.length; i++) {
        x.set(i, Math.pow(x.get(i), value));
    }
    return x;
}
