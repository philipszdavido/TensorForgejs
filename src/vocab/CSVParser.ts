import * as fs from 'fs';
import * as path from 'path';
import {VocabularyMap} from './VocabularyMap';

export class CSVParser {

    parseCsvLine(line: string): string[] {
        const result: string[] = [];
        let current = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            const nextChar = line[i + 1];

            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    current += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        return result;
    }

    async processEmailDatasetNoDeps() {
        const csvFilePath = path.resolve(__dirname, 'emails.csv');

        const fileContent = fs.readFileSync(csvFilePath, 'utf-8');

        const lines = fileContent.split(/\r?\n/);
        if (lines.length === 0 || !lines[0]) {
            throw new Error("CSV file is empty");
        }

        const headers = this.parseCsvLine(lines[0]);
        const textIdx = headers.indexOf('text');
        const labelIdx = headers.indexOf('label');

        if (textIdx === -1 || labelIdx === -1) {
            throw new Error("CSV must contain 'text' and 'label' columns in the header row.");
        }

        const vocabMap = new VocabularyMap();
        const rawTexts: string[] = [];
        const labels: number[] = [];

        console.log("Parsing CSV with native JS...");

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const columns = this.parseCsvLine(line);

            const text = columns[textIdx];
            const labelStr = columns[labelIdx];

            if (text === undefined || labelStr === undefined) continue;

            vocabMap.addDocuments(text);

            rawTexts.push(text);

            const isSpam = parseInt(labelStr, 10) === 1 ? 1 : 0;
            labels.push(isSpam);
        }

        console.log("Tokenizing vocabulary...");
        vocabMap.tokenize(5000);

        console.log("Vectorizing dataset for Neural Network...");
        const xs = rawTexts.map(text => vocabMap.vectorize(text));
        const ys = Float32Array.from(labels);

        console.log(`Successfully processed ${xs.length} emails using zero dependencies!`);
        console.log(`Ready for NN. Inputs shape: [${xs.length}, ${xs[0].length}]`);

        return {xs, ys, vocabMap};
    }

}

processEmailDatasetNoDeps().catch(err => console.error(err));
