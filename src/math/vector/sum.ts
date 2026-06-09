import assert from "../../assert/assert";
import {Vector} from "../../core/Vector";

export default function sum(vec: Vector) {
    return vec.sum();
}

// Anytime you perform a mathematical operation between two vectors of equal length where
// you pair up values according to their position in the vector (again: position 0 with 0, 1 with 1,
// and so on), it’s called an elementwise operation. Thus elementwise addition sums two vectors,
// and elementwise multiplication multiplies two vectors.

export function elementwise_multiplication(vec_a: Vector, vec_b: Vector) {
    assert(vec_a.length == vec_b.length);
    let output = Array(vec_a.length);

    for (let index = 0; index < vec_a.length; index++) {
        const a = vec_a.get(index);
        const b = vec_b.get(index);
        output[index] = a * b;
    }

    return output;

}

export function elementwise_addition(vec_a: Vector, vec_b: Vector) {
    assert(vec_a.length == vec_b.length);
    let output = Array(vec_a.length);

    for (let index = 0; index < vec_a.length; index++) {
        const a = vec_a.get(index);
        const b = vec_b.get(index);
        output[index] = a + b;
    }

    return output;
}

export function vector_sum(vec_a: Vector) {
    let sum = 0;
    for (let i = 0; i < vec_a.length; i++) {
        sum += vec_a.get(i);
    }
    return sum;
}

export function vector_subtract(vec_a: Vector, vec_b: Vector) {

    assert(vec_a.length == vec_b.length);

    const vec = new Vector(vec_a.length);
    for (let index = 0; index < vec_a.length; index++) {
        const a = vec_a.get(index);
        const b = vec_b.get(index);
        const result = a - b;
        vec.set(index, result);
    }

    return vec;
}

export function vector_average(vec_a: Vector) {
    return vector_sum(vec_a) / vec_a.length;
}

export function ele_mul(n: number, vector: Array<number>) {
    let output = Array(vector.length);

    assert(output.length == vector.length);

    for (const i of vector) {
        output[i] = n * vector[i];
    }

    return output;
}
