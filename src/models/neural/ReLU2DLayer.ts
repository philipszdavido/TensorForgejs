import {Matrix} from "../../core/Matrix";

export class ReLU2D {

    forward(input: Matrix) {

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
}
