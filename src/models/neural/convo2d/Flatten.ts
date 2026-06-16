import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {LayerInterface} from "../Types";

export class Flatten implements LayerInterface {

    private rows!: number;
    private cols!: number;

    forward(input: Matrix): Vector {

        this.rows = input.rows;
        this.cols = input.columns;

        const data: number[] = [];

        for (let r = 0; r < input.rows; r++) {
            for (let c = 0; c < input.columns; c++) {
                data.push(input.get(r, c));
            }
        }

        return Vector.from(data);
    }

    backward(grad: Vector): Matrix {

        const output =
            new Matrix(this.rows, this.cols);

        let k = 0;

        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                output.set(r, c, grad.get(k++));
            }
        }

        return output;
    }

    updateWeights() {
    }

}
