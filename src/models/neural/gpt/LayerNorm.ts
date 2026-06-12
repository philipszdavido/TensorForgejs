import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";

export default class LayerNorm {

    private eps = 1e-5;
    public gamma: Float64Array;
    public beta: Float64Array;

    constructor(public readonly hiddenSize: number) {
        this.gamma = new Float64Array(hiddenSize).fill(1.0);
        this.beta = new Float64Array(hiddenSize).fill(0.0);
    }

    forward(embedding: Matrix) {
        const rows = embedding.rows;
        const columns = embedding.columns;

        const out = Matrix.zeros(rows, columns);

        for (let i = 0; i < rows; i++) {
            const row = embedding.getRow(i);
            const vec = Vector.fromData(row)

            const normd = this.normalize(vec.toArray());
            const arr = new Float64Array(normd);
            out.setRow(i, arr);
        }

        return out
    }

    normalize(row: number[]) {

        // calc mean
        const Mean = row.reduce((a, b) => a + b, 0) / row.length;
        const Variance =
            row
                .map(v => (v - Mean) ** 2)
                .reduce((a, b) => a + b, 0)
            / row.length;

        const Std =
            Math.sqrt(Variance + this.eps);

        const normd = row.map((row, i) => {
            return ((row - Mean) / Std) * this.gamma[i] + this.beta[i];
        });

        return normd

    }
}
