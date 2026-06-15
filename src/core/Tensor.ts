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

    static fromArrayShape(
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

    add(other: Tensor): Tensor {
        if (this.data.length !== other.data.length) {
            throw new Error("Shape mismatch");
        }

        const result = new Tensor(this.shape);

        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }

        return result;
    }

    sub(other: Tensor): Tensor {
        if (this.data.length !== other.data.length) {
            throw new Error("Shape mismatch");
        }

        const result = new Tensor(this.shape);

        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }

        return result;
    }

    mul(other: Tensor): Tensor {
        if (this.data.length !== other.data.length) {
            throw new Error("Shape mismatch");
        }

        const result = new Tensor(this.shape);

        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }

        return result;
    }

    mulScalar(scalar: number): Tensor {
        const result = new Tensor(this.shape);

        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }

        return result;
    }

    addScalar(scalar: number): Tensor {
        const result = new Tensor(this.shape);

        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + scalar;
        }

        return result;
    }

    _matMul(other: Tensor): Tensor {
        const [m, n] = this.shape;
        const [n2, p] = other.shape;

        if (this.shape.length !== 2 || other.shape.length !== 2) {
            this.prettyPrint()
            other.prettyPrint("Other")
            throw new Error("matMul only supports 2D tensors");
        }

        if (n !== n2) {
            throw new Error("Shape mismatch for matMul");
        }

        const result = new Tensor([m, p]);

        const out = (result as any).data;

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {

                let sum = 0;

                for (let k = 0; k < n; k++) {
                    sum += this.data[i * n + k] * other.data[k * p + j];
                }

                out[i * p + j] = sum;
            }
        }

        return result;
    }

    matMul(other: Tensor): Tensor {

        const aDims = this.shape.length;
        const bDims = other.shape.length;

        // if (aDims === 1 && bDims === 1) {
        //     return this.dot(other);
        // }

        if (aDims === 1 && bDims === 2) {
            return this.vectorMatrixMatMul(other);
        }

        if (aDims === 2 && bDims === 1) {
            return this.matrixVectorMatMul(other);
        }

        if (aDims === 2 && bDims === 2) {
            return this.matrixMatrixMatMul(other);
        }

        throw new Error(
            `matMul not supported for shapes [${this.shape}] and [${other.shape}]`
        );
    }

    private dot(other: Tensor): number {

        const n = this.shape[0];

        if (n !== other.shape[0]) {
            throw new Error("Shape mismatch for dot product");
        }

        let sum = 0;

        for (let i = 0; i < n; i++) {
            sum += this.data[i] * other.data[i];
        }

        return sum;
    }

    private vectorMatrixMatMul(other: Tensor): Tensor {

        const n = this.shape[0];

        const rows = other.shape[0];
        const cols = other.shape[1];

        if (n !== rows) {
            throw new Error("Shape mismatch");
        }

        const result = new Tensor([cols]);

        for (let j = 0; j < cols; j++) {

            let sum = 0;

            for (let i = 0; i < n; i++) {
                sum += this.data[i] * other.get(i, j);
            }

            result.set(sum, j);
        }

        return result;
    }

    private matrixVectorMatMul(other: Tensor): Tensor {

        const rows = this.shape[0];
        const cols = this.shape[1];

        const n = other.shape[0];

        if (cols !== n) {
            throw new Error("Shape mismatch");
        }

        const result = new Tensor([rows]);

        for (let i = 0; i < rows; i++) {

            let sum = 0;

            for (let j = 0; j < cols; j++) {
                sum += this.get(i, j) * other.data[j];
            }

            result.set(sum, i);
        }

        return result;
    }

    private matrixMatrixMatMul(other: Tensor): Tensor {

        const m = this.shape[0];
        const n = this.shape[1];

        const n2 = other.shape[0];
        const p = other.shape[1];

        if (n !== n2) {
            this.prettyPrint()
            other.prettyPrint("Other")
            throw new Error("Shape mismatch");
        }

        const result = new Tensor([m, p]);

        for (let i = 0; i < m; i++) {

            for (let j = 0; j < p; j++) {

                let sum = 0;

                for (let k = 0; k < n; k++) {
                    sum +=
                        this.get(i, k) *
                        other.get(k, j);
                }

                result.set(sum, i, j);
            }
        }

        return result;
    }

    static fromArray(values: number[]): Tensor {
        const tensor = new Tensor([values.length]);

        for (let i = 0; i < values.length; i++) {
            tensor.set(values[i], i);
        }

        return tensor;
    }

    static from2DArray(values: number[][]): Tensor {
        const rows = values.length;
        const cols = values[0].length;

        const tensor = new Tensor([rows, cols]);

        let idx = 0;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                tensor.set(values[r][c], r, c);
            }
        }

        return tensor;
    }

    toArray(): number[] {
        return Array.from(this.data);
    }

    toNestedArray(): number[][] {
        if (this.shape.length !== 2) {
            throw new Error("toNestedArray only supports 2D tensors");
        }

        const [rows, cols] = this.shape;

        const result: number[][] = [];

        for (let r = 0; r < rows; r++) {
            const row: number[] = [];

            for (let c = 0; c < cols; c++) {
                row.push(this.get(r, c));
            }

            result.push(row);
        }

        return result;
    }

    prettyPrint(label?: string): void {

        if (label) console.log(`${label}`);

        const format = (value: number) =>
            value.toFixed(4).padStart(10, " ");

        const shape = this.shape;

        if (shape.length === 1) {
            let out = "[ ";
            for (let i = 0; i < shape[0]; i++) {
                out += format(this.get(i)) + " ";
            }
            out += "]";
            console.log(out);
            return;
        }

        if (shape.length === 2) {
            const [rows, cols] = shape;

            console.log(`Tensor(${rows}x${cols})`);

            for (let r = 0; r < rows; r++) {
                let row = "[ ";
                for (let c = 0; c < cols; c++) {
                    row += format(this.get(r, c)) + " ";
                }
                row += "]";
                console.log(row);
            }

            return;
        }

        console.log(`Tensor(shape=[${shape.join(", ")}])`);
        console.log(Array.from(this.data));
    }

    transpose(): Tensor {
        const t: Tensor = this;
        const [rows, cols] = t.shape;

        const result = new Tensor([cols, rows]);

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.set(t.get(i, j), j, i);
            }
        }

        return result;
    }

}
