export class VocabTextCleaner {


    clean(text: string) {

        let textArray = text.split("");

        // Force all characters to lowercase.
        textArray = this.lowercase(textArray);

        // HTML/Markup Stripping
        textArray = this.htmlStrip(textArray);

        // Punctuation & Whitespace Normalization
        textArray = this.punctuationWhitespaceNormalization(textArray)

        return textArray.join("");

    }

    lowercase(text: string[]) {
        return text.map(e => e.toLowerCase());
    }

    htmlStrip(text: string[]) {
        let isLeftTag = false;
        const collection: string[] = []

        for (let i = 0; i < text.length; i++) {
            const char = text[i]

            if (char === "<") {
                isLeftTag = true;
                continue
            } else if (isLeftTag) {
                continue
            }

            if (char === ">") {
                isLeftTag = false;
                continue;
            }

            collection.push(char);

        }

        return collection

    }

    punctuationWhitespaceNormalization(text: string[]) {

        const punctuations = ["!", ".", ":", ";", ","]

        const collection: string[] = []

        for (let i = 0; i < text.length; i++) {

            const char = text[i];
            const nextChar = text[i + 1];

            if (collection.length && collection[collection.length - 1] == " " && char === " ") {
                continue
            }

            if (punctuations.includes(char)) {
                continue
            }

            collection.push(char);

        }

        return collection
    }


}
