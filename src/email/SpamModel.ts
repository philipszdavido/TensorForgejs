import {TextFeatures, VocabTextCleaner} from "../vocab/VocabTokenizer";
import {VocabularyMap} from "../vocab/VocabularyMap";
import {EmbeddingLayer} from "../vocab/EmbeddingLayer";
import {NeuralNetworkDense} from "../models/neural/dense/NeuralNetworkDense";
import {ModelState, SequentialModel} from "../api/Sequential";
import {ActivationEnum, LossEnum} from "../models";
import {SpamAttention} from "./SpamAttention";
import softmax from "../math/softmax";

// @Todo: check for
// a memory leak,
// growing arrays that are never cleared
// accumulating gradients instead of resetting them
// excessive garbage collection
// storing intermediate values from every forward pass.

// Use average + max pooling
// compute both mean and max over the embeddings and concatenate them:
//     Embedding Sequence
//
//       │
//
// Mean Pool      Max Pool
//       │            │
// └──────┬─────┘
// │
//       Concatenate
//              │
//       Dense Network

// Embedding
//       │
// ▼
// X (T × D)
// │
// ▼
// Attention scores = XW
//       │
// ▼
// Softmax
//       │
// ▼
// Attention weights α
//       │
// ▼
// Weighted Sum
//       │
// ▼
// Dense Network
//       │
// ▼
// Loss

// Backprop:
// Loss
//  │
// ▼
// Dense backward
//  │
// ▼
// Weighted Sum backward
//  │
// ▼
// Softmax backward
//  │
// ▼
// Attention backward
//  │
// ▼
// Embedding backward

export class SpamModel {

    constructor(
        private cleaner: VocabTextCleaner,
        private vocab: VocabularyMap,
        private embedding: EmbeddingLayer,
        private network: NeuralNetworkDense,
        private embedDim: number,
        private attention: SpamAttention
    ) {
    }

    predict(text: string): number {

        const f = this.cleaner.clean(text);
        const featureVec = this.normalizeFeatures(f);
        const tokenIds = this.vocab.encode(f.text);

        const embSeq = this.embedding.forward(tokenIds);
        const attentionSeq = this.attention.forward(embSeq);

        const scores = softmax([...attentionSeq]);

        // we will mean pool scores
        const pooled = this.meanPoolScores(scores, embSeq, this.embedDim)

        const combinedInput = new Float32Array(this.embedDim + featureVec.length);
        combinedInput.set(pooled, 0);
        combinedInput.set(featureVec, this.embedDim);

        const prediction = this.network.forward(Array.from(combinedInput));
        return prediction.get(0)
    }

    train(text: string, label: number, lr: number) {
        const f = this.cleaner.clean(text);
        const featureVec = this.normalizeFeatures(f);
        const tokenIds = this.vocab.encode(f.text);

        if (tokenIds.length === 0) return;

        const embSeq = this.embedding.forward(tokenIds);

        // Attention = Embedding * Weight
        const attentionScoresSeq = this.attention.forward(embSeq);

        // Scores = softmax(Attention)
        const alphas = softmax([...attentionScoresSeq]);

        // we will mean pool scores
        const pooled = this.meanPoolScores(alphas, embSeq, this.embedDim)

        const combinedInput = new Float32Array(this.embedDim + featureVec.length);
        combinedInput.set(pooled, 0);
        combinedInput.set(featureVec, this.embedDim);

        const prediction = this.network.forward(Array.from(combinedInput));
        const target = [label];
        const denseInputGradVector = this.network.backward(target);
        const denseInputGrad = new Float32Array(denseInputGradVector.toArray());

        const embeddingGrads = denseInputGrad.subarray(0, this.embedDim);

        // dL/dalphas = gradAlphas = dL/dpool * dpool/dalphas
        // dL/dpool = embeddingGrads
        // dL/dalphas = gradAlphas = embeddingGrads * dpool/dalphas
        // pool = alphas * embedding
        // dpool/daplhas = embedding
        // dL/dalphas = embeddingGrads * embedding
        const {
            meanPoolGradEmbedding,
            gradAlphas
        } = this.meanPoolScoresBackward(embeddingGrads, tokenIds.length, this.embedDim, alphas, embSeq);

        // dL/dscores = dL/dalphas * dalphas/dscores
        // alphas = softmax(scores)
        // dalphas/dscores = softmax_derivative()
        // dL/dscores = gradAlphas * softmax_derivative

        // calc softmax backward
        const gradScores = this.softmaxBackward(gradAlphas, alphas)

        // Attention scores = Embedding * Weight
        // get grad embeddings and grad weight
        // dL/dembeddings = dL/dscores * dscores/dembeddings
        // dscores/dembeddings = weight
        // dL/dembeddings = dL/dscores * weight
        // dL/dweight = dL/dscores * dscores/dweight
        // = dL/dscores * embedding
        const gradEmbeddingsFromScores = this.attention.backward(gradScores/*, embSeq*/);

        this.embedding.backward(tokenIds, gradEmbeddingsFromScores.map((v, i) => meanPoolGradEmbedding[i] + v));

        this.network.update(lr);
        this.embedding.update(lr);
        this.attention.update(lr);

    }

    private normalizeFeatures(f: TextFeatures): number[] {
        return [
            Math.log(1 + f.urlCount),
            Math.log(1 + f.emailCount),
            Math.log(1 + f.phoneCount),
            Math.log(1 + f.moneyCount),

            f.uppercaseRatio,
            f.symbolDensity,
            f.repeatedCharScore
        ];
    }

    meanPoolScores(alphas: number[], embSeq: Float32Array, dim: number) {

        const numTokens = embSeq.length / dim;
        const result: Float32Array<ArrayBufferLike> = new Float32Array(dim);

        for (let t = 0; t < numTokens; t++) {

            const offset = t * dim;

            for (let d = 0; d < dim; d++) {
                result[d] += alphas[t] * embSeq[offset + d];
            }

        }

        return result;
    }

    meanPoolScoresBackward(gradFromDense: Float32Array, numTokens: number, dim: number, alphas: number[], embSeq: Float32Array) {

        // calc embedding grad
        const gradOut = new Float32Array(numTokens * dim);

        // calc grad alphas
        // grad_alphas = grad_from_dense * embedding
        const gradAlphas = new Float32Array(alphas.length);

        for (let t = 0; t < numTokens; t++) {
            const offset = t * dim;

            for (let d = 0; d < dim; d++) {
                gradOut[offset + d] = gradFromDense[d] * alphas[t];
                gradAlphas[t] += gradFromDense[d] * embSeq[offset + d];
            }

        }

        return {meanPoolGradEmbedding: gradOut, gradAlphas}

    }

    static prepModel(modelState: ModelState) {

        const embedding = new EmbeddingLayer(modelState.embedding!.vocabSize, modelState.embedding!.dim);
        embedding.setW(modelState.embedding!.data)

        const vocab = new VocabularyMap();
        vocab.setVocab(modelState.vocabulary!.vocab)
        vocab.setFreq(modelState.vocabulary!.frequency);

        const cleaner = new VocabTextCleaner();
        const seq = new SequentialModel(modelState)
        const network = seq.getPipeLine(0) as NeuralNetworkDense

        const attention = new SpamAttention(modelState.embedding!.dim)


        return new SpamModel(cleaner, vocab, embedding, network, modelState.embedding!.dim, attention);

    }

    generateModel() {

        const modelState: ModelState = {
            weights: {
                convWeights: [],
                denseWeights: this.network.getWeights(),
            },
            architecture: [
                {
                    type: "Dense", layers: [
                        {
                            inputSize: this.network.denseLayers[0].inputSize,
                            outputSize: 32,
                            activation: ActivationEnum.relu
                        },
                        {inputSize: 32, outputSize: 1, activation: ActivationEnum.sigmoid}
                    ]
                }
            ],
            embedding: {
                data: this.embedding.getW(),
                vocabSize: this.vocab.size(),
                dim: this.embedDim
            },
            vocabulary: {
                frequency: (() => {
                    const freq: Record<string, number> = {}
                    this.vocab.getFreq().forEach((v, k) => {
                        freq[k] = v;
                    })
                    return freq;
                })(),
                vocab: (() => {
                    const vocab: Record<string, number> = {}
                    this.vocab.getVocab().forEach((v, k) => {
                        vocab[k] = v;
                    })
                    return vocab;
                })()
            },
            loss: LossEnum.bce
        };

        return modelState;

    }

    softmaxBackward(
        gradAlpha: Float32Array,
        alpha: number[]
    ): Float32Array {

        const n = alpha.length;

        const gradScores = new Float32Array(n);

        let dot = 0;

        for (let i = 0; i < n; i++) {
            dot += gradAlpha[i] * alpha[i];
        }

        for (let i = 0; i < n; i++) {
            gradScores[i] =
                alpha[i] * (gradAlpha[i] - dot);
        }

        return gradScores;
    }

}

