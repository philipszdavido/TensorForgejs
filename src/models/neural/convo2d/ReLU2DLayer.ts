import {Matrix} from "../../../core/Matrix";
import {LayerInterface} from "../Types";

export class ReLU2D implements LayerInterface {

    private input!: Matrix;

    forward(input: Matrix) {

        this.input = input;

        const out =
            Matrix.zeros(
                input.rows,
                input.columns
            );

        for (let i = 0; i < input.rows; i++) {
            for (let j = 0; j < input.columns; j++) {

                out.set(
                    i,
                    j,
                    Math.max(
                        0,
                        input.get(i, j)
                    )
                );
            }
        }

        return out;
    }

    backward(input: Matrix) {
        const b = Matrix.zeros(input.rows, input.columns);

        for (let j = 0; j < b.rows; j++) {
            for (let k = 0; k < b.columns; k++) {
                b.set(j, k, this.input.get(j, k) > 0
                    ? input.get(j, k)
                    : 0)
            }
        }

        return b

    }

}
