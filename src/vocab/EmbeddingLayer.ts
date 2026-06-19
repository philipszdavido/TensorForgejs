export class EmbeddingLayer {

    private W: Float32Array;
    private vocabSize: number;
    private dim: number;

    private dW: Float32Array;

    constructor(vocabSize: number, dim: number) {
        this.vocabSize = vocabSize;
        this.dim = dim;

        this.W = new Float32Array(vocabSize * dim);
        this.dW = new Float32Array(vocabSize * dim);

        this.init();
    }

    private init() {
        for (let i = 0; i < this.W.length; i++) {
            this.W[i] = (Math.random() - 0.5) * 0.1;
        }
    }

    forward(tokenIds: number[]): Float32Array {

        const out = new Float32Array(tokenIds.length * this.dim);

        let outIndex = 0;

        for (let t = 0; t < tokenIds.length; t++) {

            const id = tokenIds[t];
            const row = id * this.dim;

            for (let d = 0; d < this.dim; d++) {
                out[outIndex++] = this.W[row + d];
            }
        }

        return out;
    }

    backward(tokenIds: number[], gradOut: Float32Array) {

        let outIndex = 0;

        for (let t = 0; t < tokenIds.length; t++) {

            const id = tokenIds[t];
            const row = id * this.dim;

            for (let d = 0; d < this.dim; d++) {
                this.dW[row + d] += gradOut[outIndex++];
            }
        }
    }

    update(lr: number) {
        for (let i = 0; i < this.W.length; i++) {
            this.W[i] -= lr * this.dW[i];
            this.dW[i] = 0;
        }
    }
}
