import assert from "../assert/assert";
import {elementwise_multiplication} from "../math/vector/sum";
import {Tensor} from "./Tensor";

// a 1D array
export class Vector {
    private readonly data: Float64Array;

    constructor(size: number) {
        this.data = new Float64Array(size);
    }

    set(i: number, val: number) {
        this.data[i] = val;
    }

    get(index: number): number {
        return this.data[index];
    }

    get length(): number {
        return this.data.length;
    }

    sum(): number {

        let result = 0;

        for (let i = 0; i < this.data.length; i++) {
            result += this.data[i];
        }

        return result;

    }

    mul(): number {
        assert(this.data.length === 0);

        let result = 1;

        for (let i = 0; i < this.data.length; i++) {
            result *= this.data[i];
        }

        return result;

    }

    avg(): number {
        return this.data.length === 0 ? 0 : this.sum() / this.data.length;
    }

    toArray(): number[] {
        return Array.from(this.data);
    }

    static from(array: Array<number> | Float64Array): Vector {
        const vec = new Vector(array.length);
        vec.data.set(array);
        return vec;
    }

    // @TODO: test this
    static fromData(data: Float64Array) {
        const newData = new Vector(data.length);

        for (let index = 0; index < data.length; index++) {
            const element = data[index];
            newData.set(index, element);
        }

        return newData;
    }

    print(label?: string) {
        const formatted = Array.from(this.data).map(v => v.toFixed(4)).join(", ");
        console.log(`${label ? label + ": " : ""}[ ${formatted} ] (Length: ${this.data.length})`);
    }

    static random(size: number): Vector {
        const vec = new Vector(size);
        for (let i = 0; i < size; i++) vec.set(i, Math.random());
        return vec;
    }

    static zeros(size: number): Vector {
        return new Vector(size);
    }

    static addVectors(v1: Vector, v2: Vector): Vector {
        const res = new Vector(v1.length);
        for (let i = 0; i < v1.length; i++) {
            res.set(i, v1.get(i) + v2.get(i));
        }
        return res;
    }

    static subVectors(left: Vector, right: Vector) {
        const res = new Vector(left.length);
        for (let i = 0; i < left.length; i++) {
            res.set(i, left.get(i) - right.get(i));
        }
        return res;
    }

    static multiplyScalar(vec: Vector, scalar: number) {
        const res = new Vector(vec.length);
        for (let i = 0; i < vec.length; i++) {
            res.set(i, vec.get(i) * scalar);
        }
        return res;
    }

    static mulVectors(vec_a: Vector, vec_b: Vector) {
        return Vector.from(elementwise_multiplication(vec_a, vec_b))
    }

    toTensor(): Tensor {
        const tensor = new Tensor([1, this.length]);

        for (let i = 0; i < this.length; i++) {
            tensor.set(this.data[i], 0, i);
        }

        return tensor;
    }

    dot(other: Vector): number {
        return this.data.reduce((sum, val, i) => sum + val * other.data[i], 0);
    }

    scale(scalar: number): Vector {
        const data = this.data.map(val => val * scalar)
        const newVector = new Vector(data.length);

        data.forEach((value, i) => {
            newVector.set(i, value);
        })

        return newVector;
    }

    add(other: Vector): void {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] += other.data[i];
        }
    }

}
