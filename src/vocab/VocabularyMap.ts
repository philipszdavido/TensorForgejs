import {SPAM_STOP_WORDS} from "./StopWords";

export class VocabularyMap {

    private readonly freq = new Map<string, number>();
    private readonly vocab = new Map<string, number>();
    private index = 0;
    private currentVocabSize = 1;

    constructor() {
        this.vocab.set("[UNK]", this.index)
    }

    public addDocuments(text: string): void {

        const tokens = text.split(" ");

        for (let i = 0; i < tokens.length; i++) {

            const tok = tokens[i];

            if (!tok || SPAM_STOP_WORDS.has(tok)) {
                continue;
            }

            if (this.freq.has(tok)) {
                let value = this.freq.get(tok) || 0;
                this.freq.set(tok, ++value)
            } else {
                this.freq.set(tok, 1);
            }

        }

    }

    tokenize(maxVocabSize: number = 5000) {

        const sortedTokens = Array.from(this.freq.entries())
            .sort((a, b) => b[1] - a[1])
            .map(entry => entry[0]);

        const topTokens = sortedTokens.slice(0, maxVocabSize - 1);

        this.vocab.clear();
        this.vocab.set("[UNK]", 0);

        for (let i = 0; i < topTokens.length; i++) {
            const tok = topTokens[i];
            this.vocab.set(tok, i + 1);
        }

        this.currentVocabSize = this.vocab.size;

    }

    vectorize(text: string): Float32Array {

        const tokens = text.split(" ");
        const vector = new Float32Array(this.currentVocabSize);

        for (let i = 0; i < tokens.length; i++) {
            const tok = tokens[i];

            if (!tok || SPAM_STOP_WORDS.has(tok)) {
                continue;
            }

            if (this.vocab.has(tok)) {
                const idx = this.vocab.get(tok)!;
                vector[idx] = 1.0;
            } else {
                vector[0] = 1.0;
            }
        }
        return vector;
    }

    encode(text: string): number[] {
        const tokens = text.split(/\s+/).filter(Boolean);

        const encoded: number[] = [];

        for (const tok of tokens) {
            if (this.vocab.has(tok)) {
                encoded.push(this.vocab.get(tok)!);
            } else {
                encoded.push(0);
            }
        }

        return encoded;
    }

    getVocab() {
        return this.vocab;
    }

    getFreq() {
        return this.freq;
    }

    size() {
        return this.vocab.size;
    }

}
