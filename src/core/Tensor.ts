import {Matrix} from "./Matrix";
import {Vector} from "./Vector";

export class Tensor {

    private readonly data: Float64Array;
    public readonly strides: number[];

    constructor(public readonly shape: number[]) {
        const size =
            shape.reduce(
                (a, b) => a * b,
                1
            );

        this.data =
            new Float64Array(size);

        this.strides = Tensor.computeStrides(shape);
    }

    private offset(indices: number[]): number {
        if (indices.length !== this.shape.length) {
            throw new Error("Invalid number of indices");
        }

        let index = 0;

        for (let i = 0; i < indices.length; i++) {
            index += indices[i] * this.strides[i];
        }

        return index;
    }

    get(...indices: number[]): number {
        return this.data[this.offset(indices)];
    }

    set(value: number, ...indices: number[]): void {
        this.data[this.offset(indices)] = value;
    }

    size(): number {
        return this.data.length;
    }

    static computeStrides(shape: number[]): number[] {
        const strides = new Array(shape.length);

        let stride = 1;

        for (let i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    reshape(newShape: number[]): Tensor {
        const newSize = newShape.reduce((a, b) => a * b, 1);

        if (newSize !== this.data.length) {
            throw new Error("Invalid reshape");
        }

        const t = new Tensor(newShape);

        t.data.set(this.data);

        return t;
    }

    fill(value: number) {
        this.data.fill(value);
    }

    setFloatArray(data: Float64Array) {
        for (let i = data.length - 1; i >= 0; i--) {
            this.data[i] = data[i];
        }
    }

    static zeros(shape: number[]) {
        const tensor = new Tensor(shape);
        tensor.fill(0);
        return tensor;
    }

    static fromArray(
        shape: number[],
        values: number[]
    ): Tensor {

        const tensor = new Tensor(shape);

        for (let i = 0; i < values.length; i++) {
            tensor.data[i] = values[i];
        }

        return tensor;
    }

    clone() {
        const tensor = new Tensor(this.shape);
        tensor.setFloatArray(this.data)
        return tensor;
    }

    toMatrix(): Matrix {
        const rows = this.shape[0];
        const cols = this.shape[1];
        const matrix = new Matrix(rows, cols);

        let idx = 0;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {

                matrix.set(
                    r,
                    c,
                    this.data[idx++]
                );

            }
        }

        return matrix;

    }

    static fromMatrix(
        matrix: Matrix
    ): Tensor {

        const tensor =
            new Tensor([
                matrix.rows,
                matrix.columns
            ]);

        let idx = 0;

        for (let r = 0; r < matrix.rows; r++) {
            for (let c = 0; c < matrix.columns; c++) {

                tensor.data[idx++] =
                    matrix.get(r, c);

            }
        }

        return tensor;
    }

    toVector(): Vector {

        if (this.shape.length !== 1) {
            throw new Error(
                "Tensor is not 1-dimensional"
            );
        }

        const v =
            new Vector(this.shape[0]);

        for (let i = 0; i < this.data.length; i++) {
            v.set(i, this.data[i]);
        }

        return v;
    }

}
