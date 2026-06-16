import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";
import {LayerInterface} from "../Types";

// training a convo means directly modifying the weights of your convolutional kernels
export class Convo2D implements LayerInterface {

    private feature: Matrix[] = []

    private slidingWindow: SlidingWindow[] = []

    private dKernel: Matrix[] = []
    private input!: Matrix;

    constructor(public readonly kernel: Matrix[], public readonly stride: number = 1) {
        for (let i = 0; i < this.kernel.length; i++) {
            const kernel = this.kernel[i];
            this.slidingWindow.push(new SlidingWindow(stride, kernel));
            this.dKernel.push(Matrix.zeros(kernel.rows, kernel.columns))
        }
    }

    forward(input: Matrix) {
        this.input = input;

        for (let i = 0; i < this.kernel.length; i++) {
            const kernel = this.kernel[i];
            this.feature[i] = (this.convolution(input, i));
        }

        return this.feature
    }

    convolution(input: Matrix, index: number) {

        const slide_vert = this.slidingWindow[index].slide_vert_fn(this.kernel[index].columns, input.columns);
        const slide_down = this.slidingWindow[index].slide_down_fn(this.kernel[index].rows, input.rows);

        const feature = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                const extractedWindow = this.slidingWindow[index].extractWindow(input, i * this.stride, j * this.stride);
                feature.set(i, j, this.convolveWindow(extractedWindow, index))
            }
        }

        return feature

    }

    convolveWindow(holder: Matrix, index: number) {

        let sum = 0

        for (let i = 0; i < holder.rows; i++) {
            for (let j = 0; j < holder.columns; j++) {
                const h = holder.get(i, j);
                const fill = this.kernel[index].get(i, j);
                sum += fill * h;
            }

        }

        return sum;

    }

    backward(dOutput: Matrix[]): Matrix {

        for (let index = 0; index < dOutput.length; index++) {

            // this.dKernel.fill(0);

            const dInput =
                Matrix.zeros(
                    this.input.rows,
                    this.input.columns
                );

            for (let r = 0; r < dOutput[index].rows; r++) {
                for (let c = 0; c < dOutput[index].columns; c++) {

                    const grad = dOutput[index].get(r, c);

                    const rowStart = r * this.stride;

                    const colStart = c * this.stride;

                    for (let kr = 0; kr < this.kernel[index].rows; kr++) {
                        for (let kc = 0; kc < this.kernel[index].columns; kc++) {

                            const currentKernelGrad = this.dKernel[index].get(kr, kc);

                            this.dKernel[index].set(
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
                                this.kernel[index].get(kr, kc)
                            );
                        }
                    }
                }
            }

            return dInput;

        }

        return this.input
    }

    updateWeights(lr: number) {

        for (let i = 0; i < this.kernel.length; i++) {

            const kernel = this.kernel[i];
            const dKernel = this.dKernel[i];

            for (let r = 0; r < kernel.rows; r++) {
                for (let c = 0; c < kernel.columns; c++) {

                    kernel.set(
                        r,
                        c,
                        kernel.get(r, c)
                        - lr * dKernel.get(r, c)
                    );
                }
            }

            this.zeroGrad(i)
        }

    }

    zeroGrad(index: number): void {
        this.dKernel[index].fill(0);
    }

}
