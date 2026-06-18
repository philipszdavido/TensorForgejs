import {VocabularyMap} from "../vocab/VocabularyMap";
import {CSVParser} from "../vocab/CSVParser";

export interface EmailRow {
    text: string;
    label: string;
}

export class EmailRunPipeline {

    runPipeline(absoluteOrRelativePath: string) {
        
        const parser = new CSVParser();

        const dataset = parser.parse<EmailRow>(absoluteOrRelativePath);

        const vocabMap = new VocabularyMap();
        const rawTexts: string[] = [];
        const labels: number[] = [];

        for (const row of dataset) {
            vocabMap.addDocuments(row.text);
            rawTexts.push(row.text);

            const isSpam = parseInt(row.label, 10) === 1 ? 1 : 0;
            labels.push(isSpam);
        }

        vocabMap.tokenize(5000);
        const xs = rawTexts.map(text => vocabMap.vectorize(text));
        const ys = Float32Array.from(labels);

        console.log(`Ready for NN. Inputs shape: [${xs.length}, ${xs[0].length}]`);

        return {xs, ys, vocabMap};
    }

}
