import {Matrix} from "../../core/Matrix";

export class SlidingWindow {

    constructor(private readonly stride: number, private readonly kernel: Matrix) {
    }

    extractWindow(input: Matrix, row: number, col: number) {
        const holder = Matrix.zeros(this.kernel.rows, this.kernel.columns)
        let iter_row = row;
        let iter_col = col

        for (let j = 0; j < this.kernel.rows; j++) {
            for (let k = 0; k < this.kernel.columns; k++) {
                holder.set(j, k, input.get(iter_row, iter_col))
                iter_col++
            }
            iter_col = col
            iter_row++
        }

        return holder
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
