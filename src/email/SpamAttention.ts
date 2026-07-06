import {Matrix} from "../core/Matrix";

export class SpamAttention {

    private embSeq!: Float32Array<ArrayBufferLike>

    W: Matrix;
    dW: Matrix;

    constructor(public embedDim: number) {
        this.W = Matrix.random(embedDim, 1);
        this.dW = Matrix.zeros(embedDim, 1);
    }

    forward(embSeq: Float32Array<ArrayBufferLike>) {

        this.embSeq = embSeq;

        const numTokens = embSeq.length / this.embedDim;
        const scores: Float32Array<ArrayBufferLike> = new Float32Array(numTokens); //Matrix.zeros(numTokens, 1);

        for (let t = 0; t < numTokens; t++) {

            const offset = t * this.embedDim;
            let sum = 0;

            for (let d = 0; d < this.embedDim; d++) {
                sum += embSeq[offset + d] * this.W.get(d, 0);
            }

            scores[t] = sum;

        }

        return scores

    }

    backward(gradScores: Float32Array<ArrayBufferLike>/*, embSeq: Float32Array<ArrayBufferLike>*/) {
        // dL/dweight = gradScores * embedding
        // dL/dembeddings = gradScores * weight

        this.dW.fill(0)

        const embSeq = this.embSeq;

        const numTokens = embSeq.length / this.embedDim;
        const gradEmbeddings: Float32Array<ArrayBufferLike> = new Float32Array(embSeq.length);

        for (let t = 0; t < numTokens; t++) {

            const rowIndex = t;
            const offset = t * this.embedDim;
            const s = gradScores[rowIndex];

            for (let d = 0; d < this.embedDim; d++) {
                this.dW.set(d, 0, embSeq[offset + d] * s);
                gradEmbeddings[offset + d] = s * this.W.get(d, 0);
            }

        }

        return gradEmbeddings

    }

    update(lr: number) {

        Matrix.sub(this.W, Matrix.multiplyScalar(this.dW, lr))

        this.dW.fill(0);

    }

}
