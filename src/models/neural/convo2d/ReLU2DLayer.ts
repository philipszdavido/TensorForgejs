import {Matrix} from "../../../core/Matrix";
import {LayerInterface} from "../Types";

export class ReLU2D implements LayerInterface {

    private input!: Matrix[];

    forward(input: Matrix[]) {

        this.input = input;
        const outputs = []

        for (let index = 0; index < input.length; index++) {

            const out =
                Matrix.zeros(
                    input[index].rows,
                    input[index].columns
                );

            for (let i = 0; i < input[index].rows; i++) {
                for (let j = 0; j < input[index].columns; j++) {

                    out.set(
                        i,
                        j,
                        Math.max(
                            0,
                            input[index].get(i, j)
                        )
                    );
                }
            }

            outputs.push(out);

        }

        return outputs

    }

    backward(input: Matrix[]) {

        const o = [];

        for (let index = 0; index < input.length; index++) {

            const b = Matrix.zeros(input[index].rows, input[index].columns);

            for (let j = 0; j < b.rows; j++) {
                for (let k = 0; k < b.columns; k++) {
                    b.set(j, k, this.input[index].get(j, k) > 0
                        ? input[index].get(j, k)
                        : 0)
                }
            }

            o.push(b)
        }

        return o

    }

    updateWeights() {
    }

}
