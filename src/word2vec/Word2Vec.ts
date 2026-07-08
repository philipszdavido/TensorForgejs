import {Vector} from "../core/Vector";

export class Word2Vec {

    pos_dataset: Record<string, string[]> = {}
    neg_dataset: Record<string, string[]> = {}

    contextWindow = 2
    negPair = 2

    targetEmbeddings: Record<string, Vector> = {}
    contextEmbeddings: Record<string, Vector> = {}
    embDim = 2

    learningRate = 0.025;
    epochs = 1000;

    dataset: any[][] = []

    vocab: Set<string> = new Set();

    train(corpus: string[]) {

        for (let i = 0; i < corpus.length; i++) {

            const text = corpus[i];

            const parts = text.split(' ');

            for (let j = 0; j < parts.length; j++) {

                const part = parts[j];
                this.vocab.add(part);

                const context = this.pos_skip_gram(j, parts)
                this.pos_dataset[part] = context;

                const negs = this.neg_skip_grams(j, parts);
                this.neg_dataset[part] = negs;

            }

        }

        console.log(this.pos_dataset)
        console.log(this.neg_dataset)

        this.init_embeddings();

        for (let epoch = 1; epoch <= this.epochs; epoch++) {

            // pick a target word
            // Calculate the similarity of target word and context word
            for (const targetWordKey in this.pos_dataset) {
                const ContextWords = this.pos_dataset[targetWordKey];

                for (let j = 0; j < ContextWords.length; j++) {

                    const context = ContextWords[j];

                    this.forward([this.targetEmbeddings[targetWordKey], this.contextEmbeddings[context]], 1)

                    // pick one neg
                    const negs = this.neg_dataset[targetWordKey];
                    if (negs && negs.length > 0 && negs?.[j]) {
                        this.forward([this.targetEmbeddings[targetWordKey], this.contextEmbeddings[negs[j]]], 0);
                    }

                }

            }

        }

    }

    neg_skip_grams(targetIdx: number, arr: any[]) {

        const final_negs = []

        const remove = targetIdx + this.contextWindow + 1
        const negatives = arr.slice(remove)

        final_negs.push(...negatives)

        const before = targetIdx - this.contextWindow;

        const back_negs = before <= 0 ? [] : arr.slice(0, before);

        final_negs.push(...back_negs)

        if (final_negs.length < this.negPair * this.contextWindow) {
            const start = Math.max(0, targetIdx - this.contextWindow);
            const end = Math.min(arr.length - 1, targetIdx + this.contextWindow);
            const forbidden = new Set(arr.slice(start, end + 1));

            const pool = Array.from(this.vocab).filter(word => !forbidden.has(word));
            let i = 0;

            for (let j = final_negs.length; j < this.negPair; j++) {
                final_negs.push(pool[i]);
                i++
            }

        }

        return final_negs
    }

    pos_skip_gram(index: number, arr: any[]) {

        const after = arr.slice(index + 1, this.contextWindow + index + 1);

        const before = index - this.contextWindow >= 0 ? arr.slice(Math.abs(index - this.contextWindow), Math.abs(index - this.contextWindow) + this.contextWindow) : [];

        return [...after, ...before];

    }

    init_embeddings() {

        for (const posDatasetKey in this.pos_dataset) {
            this.targetEmbeddings[posDatasetKey] = Vector.random(this.embDim)
            this.targetEmbeddings[posDatasetKey]!.print(posDatasetKey)
        }

        for (const posDatasetKey in this.neg_dataset) {
            this.contextEmbeddings[posDatasetKey] = Vector.random(this.embDim)
        }

    }

    forward(weights: Vector[], label: number) {

        if (weights.length < 2) {
            return;
        }

        // the target word embedding
        const w1 = weights[0];
        // context(positive or negative) embedding
        const w2 = weights[1];

        const dotProduct = w1.dot(w2);

        const prediction = 1 / (1 + Math.exp(-dotProduct));

        const error = prediction - label;

        const learningRate = 0.01;

        // Gradients:
        // dL/dw1 = error * w2
        // dL/dw2 = error * w1

        const w1Update = w2.scale(-learningRate * error);
        const w2Update = w1.scale(-learningRate * error);

        w1.add(w1Update);
        w2.add(w2Update);
    }

    similarity(word1: string, word2: string): number {
        const v1 = this.targetEmbeddings[word1.toLowerCase()];
        const v2 = this.targetEmbeddings[word2.toLowerCase()];
        if (!v1 || !v2) return 0;

        const dot = v1.dot(v2);
        const mag1 = Math.sqrt(v1.dot(v1));
        const mag2 = Math.sqrt(v2.dot(v2));

        return dot / (mag1 * mag2);
    }

}
