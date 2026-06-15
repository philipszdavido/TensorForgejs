import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";

// training a convo means directly modifying the weights of your convolutional kernels
export class Convo2D {

    private feature!: Matrix

    private slidingWindow: SlidingWindow

    constructor(public readonly kernel: Matrix, public readonly stride: number = 1) {
        this.slidingWindow = new SlidingWindow(stride, kernel)
    }

    forward(input: Matrix) {
        return this.convolution(input);
    }

    convolution(input: Matrix) {

        const slide_vert = this.slidingWindow.slide_vert_fn(this.kernel.columns, input.columns);
        const slide_down = this.slidingWindow.slide_down_fn(this.kernel.rows, input.rows);

        this.feature = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                const extractedWindow = this.slidingWindow.extractWindow(input, i * this.stride, j * this.stride);
                this.feature.set(i, j, this.convolveWindow(extractedWindow))
            }
        }

        return this.feature

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

    backward() {

    }


}
