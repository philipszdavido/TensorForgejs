import {Matrix} from "../core/Matrix";

export default function transpose(matrix: Matrix): Matrix {

    const result = new Matrix(
        matrix.rows,
        matrix.columns
    );

    for (let i = 0; i < matrix.rows; i++) {
        for (let j = 0; j < matrix.columns; j++) {
            result.set(j, i, matrix.get(i, j));
        }
    }

    return result;
}
