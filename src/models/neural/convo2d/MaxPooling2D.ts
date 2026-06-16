import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";
import {LayerInterface} from "../Types";

export class MaxPooling2D implements LayerInterface {

    private feature: Matrix[] = []
    private slidingWindow: SlidingWindow[] = []
    private input!: Matrix[];
    private maxPositionsCache: Map<string, { row: number; col: number }>[] = [];

    constructor(public readonly kernel: Matrix[], public readonly stride: number = 1) {

        for (let i = 0; i < this.kernel.length; i++) {

            const kernel = this.kernel[i];

            this.slidingWindow[i] = new SlidingWindow(stride, kernel)

        }

    }

    forward(input: Matrix[]) {

        this.input = input;

        for (let i = 0; i < this.kernel.length; i++) {

            this.maxPool(input[i], i);
        }

        return this.feature;

    }

    maxPool(input: Matrix, index: number) {
        const slide_vert = this.slidingWindow[index].slide_vert_fn(this.kernel[index].columns, input.columns);
        const slide_down = this.slidingWindow[index].slide_down_fn(this.kernel[index].rows, input.rows);
        this.maxPositionsCache[index] = new Map();
        this.feature[index] = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                const extractedWindow = this.slidingWindow[index].extractWindow(input, i * this.stride, j * this.stride);
                const result = this.max(extractedWindow)
                this.feature[index].set(i, j, result.max)

                this.maxPositionsCache[index].set(`${i},${j}`, {
                    row: i * this.stride + result.localRow,
                    col: j * this.stride + result.localCol
                });
            }
        }
        return this.feature
    }

    max(extractedWindow: Matrix) {
        let localRow = 0;
        let localCol = 0;
        let max = -Infinity

        for (let i = 0; i < extractedWindow.rows; i++) {
            for (let j = 0; j < extractedWindow.columns; j++) {
                let v = extractedWindow.get(i, j);
                if (v > max) {
                    localRow = i;
                    localCol = j;
                    max = v;
                }
            }
        }

        return {max, localRow, localCol};

    }

    backward(output: Matrix[]) {

        const inputs = []

        for (let index = 0; index < output.length; index++) {

            const input = Matrix.zeros(this.input[index].rows, this.input[index].columns);

            for (let i = 0; i < output[index].rows; i++) {
                for (let j = 0; j < output[index].columns; j++) {
                    const grad = output[index].get(i, j);
                    const pos = this.maxPositionsCache[index].get(`${i},${j}`);

                    if (pos) {
                        const currentGrad = input.get(pos.row, pos.col);
                        input.set(pos.row, pos.col, currentGrad + grad);
                    }
                }
            }

            inputs.push(input);

        }

        return inputs

    }

    updateWeights() {
    }

}
