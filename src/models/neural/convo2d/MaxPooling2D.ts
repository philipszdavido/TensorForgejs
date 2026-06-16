import {Matrix} from "../../../core/Matrix";
import {SlidingWindow} from "./SlidingWindow";
import {LayerInterface} from "../Types";

export class MaxPooling2D implements LayerInterface {

    private feature!: Matrix
    private slidingWindow: SlidingWindow
    private input!: Matrix;
    private maxPositionsCache!: Map<string, { row: number; col: number }>;

    constructor(public readonly kernel: Matrix, public readonly stride: number = 1) {
        this.slidingWindow = new SlidingWindow(stride, kernel)
    }

    forward(input: Matrix) {
        this.input = input;
        return this.maxPool(input);
    }

    maxPool(input: Matrix) {
        const slide_vert = this.slidingWindow.slide_vert_fn(this.kernel.columns, input.columns);
        const slide_down = this.slidingWindow.slide_down_fn(this.kernel.rows, input.rows);
        this.maxPositionsCache = new Map();
        this.feature = new Matrix(slide_down, slide_vert);

        // scan the kernel throughout the input
        for (let i = 0; i < slide_down; i++) {
            for (let j = 0; j < slide_vert; j++) {
                const extractedWindow = this.slidingWindow.extractWindow(input, i * this.stride, j * this.stride);
                const result = this.max(extractedWindow)
                this.feature.set(i, j, result.max)

                this.maxPositionsCache.set(`${i},${j}`, {
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

    backward(output: Matrix) {

        const input = Matrix.zeros(this.input.rows, this.input.columns);

        for (let i = 0; i < output.rows; i++) {
            for (let j = 0; j < output.columns; j++) {
                const grad = output.get(i, j);
                const pos = this.maxPositionsCache.get(`${i},${j}`);

                if (pos) {
                    const currentGrad = input.get(pos.row, pos.col);
                    input.set(pos.row, pos.col, currentGrad + grad);
                }
            }
        }

        return input;

    }

    updateWeights() {
    }

}
