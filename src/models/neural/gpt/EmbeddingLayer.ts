import {Matrix} from "../../../core/Matrix";

// vocab size X embedding dimension
export default class EmbeddingLayer {

    embeddings!: Matrix

    constructor(public readonly vocab_size: number, public readonly embedding_size: number) {
        this.initialization(vocab_size, embedding_size);
    }

    initialization(vocab_size: number, embedding_size: number) {
        this.embeddings = Matrix.random(vocab_size, embedding_size);
    }

    lookup(token_ids: number[]): Matrix {

        const result = Matrix.zeros(
            token_ids.length,
            this.embedding_size
        );

        for (let i = 0; i < token_ids.length; i++) {

            const token = token_ids[i];

            result.setRow(
                i,
                this.embeddings.getRow(token)
            );
        }

        return result;
    }

}
