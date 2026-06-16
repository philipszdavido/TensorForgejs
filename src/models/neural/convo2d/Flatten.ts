import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {LayerInterface} from "../Types";

export class Flatten implements LayerInterface {
    private inputShape: { rows: number, cols: number }[] = [];

    forward(input: Matrix[]): Vector {
        this.inputShape = input.map(m => ({rows: m.rows, cols: m.columns}));

        const flattenedData: number[] = [];

        for (const matrix of input) {
            for (let r = 0; r < matrix.rows; r++) {
                for (let c = 0; c < matrix.columns; c++) {
                    flattenedData.push(matrix.get(r, c));
                }
            }
        }

        return Vector.from(flattenedData);
    }

    backward(grad: Vector): Matrix[] {
        const output: Matrix[] = [];
        let cursor = 0;

        for (const shape of this.inputShape) {
            const matrix = new Matrix(shape.rows, shape.cols);
            for (let r = 0; r < shape.rows; r++) {
                for (let c = 0; c < shape.cols; c++) {
                    matrix.set(r, c, grad.get(cursor++));
                }
            }
            output.push(matrix);
        }

        return output;
    }

    updateWeights() {
    }
}
