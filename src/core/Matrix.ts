import {Vector} from "./Vector";
import assert from "../assert/assert";
import {Tensor} from "./Tensor";

// 2d array
export class Matrix {
    private readonly data: Float64Array;

    constructor(
        public readonly rows: number,
        public readonly columns: number
    ) {
        this.data = new Float64Array(rows * columns);
    }

    get(r: number, c: number): number {
        return this.data[r * this.columns + c];
    }

    set(r: number, c: number, val: number) {
        this.data[r * this.columns + c] = val;
    }

    getRow(r: number): Float64Array {
        const start = r * this.columns;
        return this.data.subarray(
            start,
            start + this.columns
        );
    }

    setRow(r: number, row: Float64Array) {

        const start = r * this.columns;

        this.data.set(row, start);
    }

    static matrixMulVector(m: Matrix, v: Vector): Vector {
        assert(m.columns === v.length, `Shape mismatch: Matrix columns (${m.columns}) must equal Vector length (${v.length})`);
        const result = new Vector(m.rows);

        for (let i = 0; i < m.rows; i++) {
            let sum = 0;
            const rowOffset = i * m.columns;
            for (let j = 0; j < m.columns; j++) {
                sum += m.data[rowOffset + j] * v.get(j);
            }
            result.set(i, sum);
        }
        return result;
    }

    static outerProduct(a: Vector, b: Vector): Matrix {
        const result = new Matrix(a.length, b.length);
        for (let i = 0; i < a.length; i++) {
            const rowOffset = i * b.length;
            const aval = a.get(i);
            for (let j = 0; j < b.length; j++) {
                result.data[rowOffset + j] = aval * b.get(j);
            }
        }
        return result;
    }

    static zeros(rows: number, columns: number): Matrix {

        const data = new Matrix(rows, columns);

        for (let index = 0; index < rows; index++) {
            for (let j = 0; j < columns; j++) {
                data.set(index, j, 0);
            }
        }

        return data

    }

    print(label?: string) {
        if (label) console.log(`\n${label}`);
        console.log(`Matrix(${this.rows}x${this.columns})`);
        for (let i = 0; i < this.rows; i++) {
            let rowStr = "[ ";
            for (let j = 0; j < this.columns; j++) {
                rowStr += this.get(i, j).toFixed(4) + " ";
            }
            console.log(rowStr + "]");
        }
    }

    static add(m: Matrix, v: Matrix) {

        // @TODO: test this
        assert(m.columns === v.columns, `Shape mismatch: Matrix columns (${m.columns}) must equal Matrix columns (${v.columns})`);

        const result = new Matrix(m.rows, m.columns);

        for (let r = 0; r < m.rows; r++) {
            for (let c = 0; c < m.columns; c++) {
                const dVal = v.get(r, c);

                result.set(r, c, m.get(r, c) + dVal);
            }
        }

        return result;

    }

    static sub(left: Matrix, right: Matrix) {

        // @TODO: test this
        assert(left.columns === right.columns, `Shape mismatch: Matrix columns (${left.columns}) must equal Matrix columns (${left.columns})`);

        const result = new Matrix(right.rows, right.columns);

        for (let r = 0; r < left.rows; r++) {
            for (let c = 0; c < left.columns; c++) {
                const rVal = right.get(r, c);

                result.set(r, c, left.get(r, c) - rVal);
            }
        }

        return result;

    }

    // @TODO: test this
    // [[0,1,2,3,4,5]]
    static from(arr: Array<Array<number>>) {
        const data = new Matrix(arr.length, arr[0].length);

        for (let index = 0; index < arr.length; index++) {
            const col = arr[index];
            for (let j = 0; j < col.length; j++) {
                data.set(index, j, col[j]);
            }
        }

        return data;
    }

    // @TODO: test this
    static fromVector(vec: Vector) {
        const data = new Matrix(vec.length, 1);
        for (let index = 0; index < vec.length; index++) {
            data.set(0, index, vec.get(index));
        }

        return data;
    }

    static random(rows: number, columns: number) {
        const data = new Matrix(rows, columns);

        for (let index = 0; index < rows; index++) {
            for (let j = 0; j < columns; j++) {
                data.set(index, j, Math.random());
            }
        }

        return data;
    }

    // @TODO: test this
    mul(a: Matrix) {
        const newData = new Matrix(a.rows, this.columns);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < a.columns; j++) {
                let sum = 0;
                for (let k = 0; k < this.columns; k++) {
                    const result = this.get(i, j) * a.get(i, j);
                    sum += result;
                }

                newData.set(i, j, sum);
            }
        }
        return newData;
    }

    // @TODO: test this
    static batchMul(X: Matrix, W: Matrix): Matrix {

        assert(X.columns === W.rows, "Shape mismatch: Batch × Weights");

        const result = new Matrix(X.rows, W.columns);

        for (let i = 0; i < X.rows; i++) {
            for (let j = 0; j < W.columns; j++) {
                let sum = 0;

                for (let k = 0; k < X.columns; k++) {
                    sum += X.get(i, k) * W.get(k, j);
                }

                result.set(i, j, sum);
            }
        }

        return result;
    }

    toNestedArray(): number[][] {
        const result: number[][] = [];
        for (let r = 0; r < this.rows; r++) {
            const rowData = Array.from(this.getRow(r));
            result.push(rowData);
        }
        return result;
    }

    static multiplyScalar(matrix: Matrix, scalar: number): Matrix {
        const result = new Matrix(matrix.rows, matrix.columns);

        for (let r = 0; r < matrix.rows; r++) {
            for (let c = 0; c < matrix.columns; c++) {
                result.set(r, c, matrix.get(r, c) * scalar);
            }
        }

        return result;
    }

    copy() {

        const c = new Matrix(this.rows, this.columns);

        for (let ii = 0; ii < this.rows; ii++) {
            for (let jj = 0; jj < this.columns; jj++) {
                c.set(ii, jj, this.get(ii, jj));
            }
        }

        return c;

    }

    toTensor(): Tensor {
        const tensor = new Tensor([this.rows, this.columns])
        for (let i = 0; i < this.data.length; i++) {
            tensor.set(i, this.data[i]);
        }
        return tensor;
    }

}
