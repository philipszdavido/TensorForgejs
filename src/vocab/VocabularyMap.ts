import {VocabTextCleaner} from "./VocabTokenizer";

export class VocabularyMap {
    private readonly cleaner = new VocabTextCleaner()

    private readonly vocab = new Map<string, number>();

    tokenize(text: string) {
        const tokens = text.split(" ");

        for (let i = 0; i < tokens.length; i++) {
            const tok = tokens[i];
            if (this.vocab.has(tok)) {
                let value = this.vocab.get(tok) || 0;
                this.vocab.set(tok, ++value)
            } else {
                this.vocab.set(tok, 1);
            }
        }

    }

    encode(text: string) {
        
    }

    getVocab() {
        return this.vocab;
    }
}
