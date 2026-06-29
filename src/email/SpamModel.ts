import {TextFeatures, VocabTextCleaner} from "../vocab/VocabTokenizer";
import {VocabularyMap} from "../vocab/VocabularyMap";
import {EmbeddingLayer} from "../vocab/EmbeddingLayer";
import {NeuralNetworkDense} from "../models/neural/dense/NeuralNetworkDense";
import {ModelState, SequentialModel} from "../api/Sequential";
import {ActivationEnum, LossEnum} from "../models";

export class SpamModel {

    constructor(
        private cleaner: VocabTextCleaner,
        private vocab: VocabularyMap,
        private embedding: EmbeddingLayer,
        private network: NeuralNetworkDense,
        private embedDim: number
    ) {
    }

    predict(text: string): number {

        const f = this.cleaner.clean(text);
        const featureVec = this.normalizeFeatures(f);
        const tokenIds = this.vocab.encode(f.text);

        const embSeq = this.embedding.forward(tokenIds);
        const pooled = this.meanPool(embSeq, this.embedDim);

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
        const pooled = this.meanPool(embSeq, this.embedDim);

        const combinedInput = new Float32Array(this.embedDim + featureVec.length);
        combinedInput.set(pooled, 0);
        combinedInput.set(featureVec, this.embedDim);

        const prediction = this.network.forward(Array.from(combinedInput));
        const target = [label];
        const denseInputGradVector = this.network.backward(target);
        const denseInputGrad = new Float32Array(denseInputGradVector.toArray());

        const embeddingGrads = denseInputGrad.subarray(0, this.embedDim);

        const gradToEmbedding = this.meanPoolBackward(embeddingGrads, tokenIds.length, this.embedDim);
        this.embedding.backward(tokenIds, gradToEmbedding);

        this.network.update(lr);
        this.embedding.update(lr);
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

    private _meanPool(x: Float32Array, dim: number): number[] {

        if (x.length === 0) {
            return new Array(dim).fill(0);
        }

        const steps = x.length / dim;
        const out = new Array(dim).fill(0);

        for (let i = 0; i < steps; i++) {
            for (let d = 0; d < dim; d++) {
                out[d] += x[i * dim + d];
            }
        }

        for (let d = 0; d < dim; d++) {
            out[d] /= steps || 1;
        }

        return out;
    }

    private bceGrad(pred: number, label: number): number {
        const eps = 1e-7;
        const p = Math.min(Math.max(pred, eps), 1 - eps);
        return -(label / p) + (1 - label) / (1 - p);
    }

    private expandGrad(
        grad: number[],
        seqLen: number,
        dim: number
    ): Float32Array {

        const out = new Float32Array(seqLen * dim);

        for (let i = 0; i < seqLen; i++) {
            for (let d = 0; d < dim; d++) {
                out[i * dim + d] = grad[d] / seqLen;
            }
        }

        return out;
    }

    public meanPool(embSeq: Float32Array, dim: number): Float32Array {
        const pooled = new Float32Array(dim);
        const numTokens = embSeq.length / dim;

        if (numTokens === 0) return pooled;

        for (let t = 0; t < numTokens; t++) {
            const offset = t * dim;
            for (let d = 0; d < dim; d++) {
                pooled[d] += embSeq[offset + d];
            }
        }

        for (let d = 0; d < dim; d++) {
            pooled[d] /= numTokens;
        }

        return pooled;
    }

    public meanPoolBackward(gradFromDense: Float32Array, numTokens: number, dim: number): Float32Array {

        const gradOut = new Float32Array(numTokens * dim);

        if (numTokens === 0) return gradOut;

        for (let t = 0; t < numTokens; t++) {
            const offset = t * dim;
            for (let d = 0; d < dim; d++) {
                gradOut[offset + d] = gradFromDense[d] / numTokens;
            }
        }

        return gradOut;
    }

    static prepModel(modelState: ModelState) {

        const embedding = new EmbeddingLayer(modelState.embedding!.vocabSize, modelState.embedding!.dim);
        embedding.setW(modelState.embedding!.data)

        const vocab = new VocabularyMap();
        vocab.setVocab(modelState.vocabularyMap!.vocab)
        vocab.setFreq(modelState.vocabularyMap!.frequency);

        const cleaner = new VocabTextCleaner();
        const seq = new SequentialModel(modelState)
        const network = seq.getPipeLine(0) as NeuralNetworkDense


        return new SpamModel(cleaner, vocab, embedding, network, modelState.embedding!.dim);

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
            vocabularyMap: {
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

}
