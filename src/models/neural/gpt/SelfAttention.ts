import {Matrix} from "../../../core/Matrix";

export default class SelfAttention {

    Wq: Matrix;
    Wk: Matrix;
    Wv: Matrix;

    dk: number;

    constructor(public readonly dModel: number) {
        this.dk = dModel;

        // @TODO: use Xavier or He to update
        this.Wq = Matrix.random(dModel, dModel);
        this.Wk = Matrix.random(dModel, dModel);
        this.Wv = Matrix.random(dModel, dModel);
    }

    forward(x: Matrix): Matrix {
        const T = x.rows;
        const D = x.columns;

        const Q = this.project(x, this.Wq);
        const K = this.project(x, this.Wk);
        const V = this.project(x, this.Wv);

        const scores = Matrix.zeros(T, T);

        for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {

                let dot = 0;
                for (let k = 0; k < D; k++) {
                    dot += Q.get(i, k) * K.get(j, k);
                }

                scores.set(i, j, dot / Math.sqrt(this.dk));
            }
        }

        const attn = Matrix.zeros(T, T);

        for (let i = 0; i < T; i++) {
            const row = new Float64Array(T);

            let max = -Infinity;
            for (let j = 0; j < T; j++) {
                max = Math.max(max, scores.get(i, j));
            }

            let sum = 0;
            for (let j = 0; j < T; j++) {
                row[j] = Math.exp(scores.get(i, j) - max);
                sum += row[j];
            }

            for (let j = 0; j < T; j++) {
                row[j] /= sum;
                attn.set(i, j, row[j]);
            }
        }

        const output = Matrix.zeros(T, D);

        for (let i = 0; i < T; i++) {
            for (let d = 0; d < D; d++) {

                let sum = 0;

                for (let j = 0; j < T; j++) {
                    sum += attn.get(i, j) * V.get(j, d);
                }

                output.set(i, d, sum);
            }
        }

        return output;
    }

    private project(x: Matrix, W: Matrix): Matrix {
        const T = x.rows;
        const D = W.columns;

        const out = Matrix.zeros(T, D);

        for (let i = 0; i < T; i++) {
            for (let j = 0; j < D; j++) {

                let sum = 0;
                for (let k = 0; k < x.columns; k++) {
                    sum += x.get(i, k) * W.get(k, j);
                }

                out.set(i, j, sum);
            }
        }

        return out;
    }
}
