import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";

export class MaxPooling2D {

    private feature!: Matrix
    private slidingWindow: SlidingWindow

    constructor(public readonly kernel: Matrix, public readonly stride: number = 1) {
        this.slidingWindow = new SlidingWindow(stride, kernel)
    }

    forward(input: Matrix) {
        return this.maxPool(input);
    }

    maxPool(input: Matrix) {
        const slide_vert = this.slidingWindow.slide_vert_fn(this.kernel.columns, input.columns);
        const slide_down = this.slidingWindow.slide_down_fn(this.kernel.rows, input.rows);

        this.feature = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                const extractedWindow = this.slidingWindow.extractWindow(input, i * this.stride, j * this.stride);
                this.feature.set(i, j, this.max(extractedWindow))
            }
        }
        return this.feature
    }

    max(extractedWindow: Matrix) {

        let array = []

        for (let i = 0; i < extractedWindow.rows; i++) {
            for (let j = 0; j < extractedWindow.columns; j++) {
                array.push(extractedWindow.get(i, j));
            }
        }

        let max = -Infinity;

        for (const v of array) {
            if (v > max) {
                max = v;
            }
        }

        return max;
    }

}
