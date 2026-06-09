// a 1D array

import {Matrix} from "./Matrix";
import assert from "../assert/assert";

export class Vector {
    private readonly data: Float32Array;

    constructor(size: number) {
        this.data = new Float32Array(size);
    }

    set(i: number, data: number) {
        this.data[i] = data;
    }

    get(index: number) {
        return this.data[index];
    }

    get length() {
        return this.data.length;
    }

    sum(): number {
        let result = 0;
        for (let index = 0; index < this.data.length; index++) {
            const element = this.data[index];
            if (index == 0) {
                result = element;
                continue;
            }

            result += element;
        }
        return result;
    }

    mul(): number {
        let result = 0;
        for (let index = 0; index < this.data.length; index++) {
            const element = this.data[index];
            if (index == 0) {
                result = element;
                continue;
            }
            result *= element;
        }
        return result;
    }

    avg(): number {
        let result = 0;
        for (let index = 0; index < this.data.length; index++) {
            const element = this.data[index];
            if (index == 0) {
                result = element;
                continue;
            }

            result += element;
        }
        return result / this.length;
    }

    toArray() {
        return [...this.data]
    }

    static fromData(data: Float32Array) {
        const newData = new Vector(data.length);

        for (let index = 0; index < data.length; index++) {
            const element = data[index];
            newData.set(index, element);
        }

        return newData;
    }

    static from(array: Array<number>): Vector {
        const newData = new Vector(array.length);

        for (let index = 0; index < array.length; index++) {
            const element = array[index];
            newData.set(index, element);
        }

        return newData;
    }

    static vectorMulMatrix(v: Vector, m: Matrix): Vector {

        assert(v.length === m.rows, "Shape mismatch: Vector × Matrix");

        const result = new Vector(m.columns);

        for (let j = 0; j < m.columns; j++) {
            let sum = 0;

            for (let i = 0; i < m.rows; i++) {
                sum += v.get(i) * m.get(i, j);
            }

            result.set(j, sum);
        }

        return result;
    }

    print(label?: string) {
        console.log(`Vector(${this.data.length})`);

        const formatted = Array.from(this.data)
            .map(v => v.toFixed(4))
            .join(", ");

        if (label) {
            console.log(`${label}: [ ${formatted} ]`);
        } else {
            console.log(`[ ${formatted} ]`);
        }
    }

    static random(size: number) {
        const vec = new Vector(size);
        for (let i = 0; i < size; i++) {
            vec.set(i, Math.random());
        }
        return vec;
    }

    static zeros(size: number) {
        const vec = new Vector(size);
        for (let i = 0; i < size; i++) {
            vec.set(i, 0);
        }
        return vec;
    }
}
