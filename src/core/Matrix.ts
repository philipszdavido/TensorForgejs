import {Vector} from "./Vector";
import assert from "../assert/assert";

// 2d array
export class Matrix {
    private readonly data: Array<Vector> = [];

    constructor(
        public readonly columns: number,
        public readonly rows: number,
    ) {
        for (let index = 0; index < this.rows; index++) {
            const row = new Vector(this.columns);

            for (let j = 0; j < this.columns; j++) {
                row.set(j, j);
            }

            this.data.push(row);
        }
    }

    get(r: number, c: number) {
        return this.data[r].get(c);
    }

    set(r: number, c: number, data: number) {
        this.data[r].set(c, data);
    }

    mul(a: Matrix) {
        const newData = new Matrix(a.columns, this.rows);

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

    toVector(): Vector {
        return this.data[0];
    }

    static matrixMulVector(m: Matrix, v: Vector): Vector {
        assert(m.columns === v.length, `Shape mismatch: Matrix columns (${m.columns}) !== Vector length (${v.length})`);

        const result = new Vector(m.rows);

        for (let i = 0; i < m.rows; i++) {
            let sum = 0;
            for (let j = 0; j < m.columns; j++) {
                sum += m.get(i, j) * v.get(j);
            }
            result.set(i, sum);
        }

        return result;
    }

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

    static fromVector(vec: Vector) {
        const data = new Matrix(vec.length, 1);
        for (let index = 0; index < vec.length; index++) {
            data.set(0, index, vec.get(index));
        }

        return data;
    }

    static random(rows: number, columns: number) {
        const data = new Matrix(columns, rows);

        for (let index = 0; index < rows; index++) {
            for (let j = 0; j < columns; j++) {
                data.set(index, j, Math.random());
            }
        }

        return data;
    }

    static zeros(rows: number, columns: number) {
        const data = new Matrix(columns, rows);

        for (let index = 0; index < rows; index++) {
            for (let j = 0; j < columns; j++) {
                data.set(index, j, 0);
            }
        }

        return data;
    }

    print(label?: string) {
        if (label) console.log(`\n${label}`);

        console.log(`Matrix(${this.rows}x${this.columns})`);

        for (let i = 0; i < this.rows; i++) {
            let rowStr = "[ ";

            for (let j = 0; j < this.columns; j++) {
                rowStr += this.get(i, j).toFixed(4) + " ";
            }

            rowStr += "]";
            console.log(rowStr);
        }
    }

}
