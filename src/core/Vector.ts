import assert from "../assert/assert";

// a 1D array
export class Vector {
    public readonly data: Float64Array; // Upgrade to 64-bit precision matching Claude edition

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
    static fromData(data: Float32Array) {
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
}
