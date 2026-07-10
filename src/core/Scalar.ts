import {Vector} from "./Vector";

export class Scalar {

    constructor(private value: number) {
    }

    static get one() {
        return (new Scalar(1));
    }

    sub(vec: Vector) {
        for (let i = 0; i < vec.length; i++) {
            vec.set(i, this.value - vec.get(i));
        }
        return vec;
    }
}
