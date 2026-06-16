import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";
import {LayerInterface} from "../Types";

// training a convo means directly modifying the weights of your convolutional kernels
export class Convo2D implements LayerInterface {

    private feature!: Matrix

    private slidingWindow: SlidingWindow

    private dKernel: Matrix
    private input!: Matrix;

    constructor(public readonly kernel: Matrix, public readonly stride: number = 1) {
        this.slidingWindow = new SlidingWindow(stride, kernel);
        this.dKernel = Matrix.zeros(kernel.rows, kernel.columns)
    }

    forward(input: Matrix) {
        this.input = input;
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

    backward(dOutput: Matrix): Matrix {

        // this.dKernel.fill(0);

        const dInput =
            Matrix.zeros(
                this.input.rows,
                this.input.columns
            );

        for (let r = 0; r < dOutput.rows; r++) {
            for (let c = 0; c < dOutput.columns; c++) {

                const grad = dOutput.get(r, c);

                const rowStart = r * this.stride;

                const colStart = c * this.stride;

                for (let kr = 0; kr < this.kernel.rows; kr++) {
                    for (let kc = 0; kc < this.kernel.columns; kc++) {

                        const currentKernelGrad = this.dKernel.get(kr, kc);

                        this.dKernel.set(
                            kr,
                            kc,
                            currentKernelGrad +
                            grad *
                            this.input.get(
                                rowStart + kr,
                                colStart + kc
                            )
                        );

                        const currentInputGrad =
                            dInput.get(
                                rowStart + kr,
                                colStart + kc
                            );

                        dInput.set(
                            rowStart + kr,
                            colStart + kc,
                            currentInputGrad +
                            grad *
                            this.kernel.get(kr, kc)
                        );
                    }
                }
            }
        }

        return dInput;
    }

    updateWeights(lr: number) {

        for (let r = 0; r < this.kernel.rows; r++) {
            for (let c = 0; c < this.kernel.columns; c++) {

                this.kernel.set(
                    r,
                    c,
                    this.kernel.get(r, c)
                    - lr * this.dKernel.get(r, c)
                );
            }
        }

        this.zeroGrad()

    }

    zeroGrad(): void {
        this.dKernel.fill(0);
    }

}
