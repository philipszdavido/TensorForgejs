import {Matrix} from "../../core/Matrix";

export class Convo2D {

    private readonly holder: Matrix
    private feature!: Matrix

    constructor(public readonly kernel: Matrix, public readonly stride: number = 1) {
        this.holder = Matrix.zeros(kernel.rows, kernel.columns)
    }

    forward(input: Matrix) {
        this.convolution(input);
    }

    convolution(input: Matrix) {

        const slide_vert = this.slide_vert_fn(this.kernel.columns, input.columns);
        const slide_down = this.slide_down_fn(this.kernel.rows, input.rows);

        this.feature = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                this.extractWindow(input, i * this.stride, j * this.stride);
                this.feature.set(i, j, this.convolveWindow(this.holder))
            }
        }

    }

    convolveWindow(holder: Matrix) {

        let sum = 0

        for (let i = 0; i < holder.rows; i++) {
            for (let j = 0; j < holder.columns; j++) {
                const h = holder.get(i, j);
                const fill = this.kernel.get(i, j);
                sum += fill * h;
            }

        }

        return sum;

    }

    extractWindow(input: Matrix, row: number, col: number) {
        let iter_row = row;
        let iter_col = col

        for (let j = 0; j < this.kernel.rows; j++) {
            for (let k = 0; k < this.kernel.columns; k++) {
                this.holder.set(j, k, input.get(iter_row, iter_col))
                iter_col++
            }
            iter_col = col
            iter_row++
        }
    }

    slide_vert_fn(col1: number, col2: number) {
        let acc = 0

        let temp = col2

        for (let i = 0; i < col2; i = i + this.stride) {
            const diff = temp - col1
            if (diff < 0) break
            temp = temp - this.stride
            acc += 1
        }
        return acc;
    }

    slide_down_fn(row1: number, row2: number) {
        let acc = 0

        let temp = row2

        for (let i = 0; i < row2; i = i + this.stride) {
            const diff = temp - row1
            if (diff < 0) break
            temp = temp - this.stride
            acc += 1
        }
        return acc;

    }

}
