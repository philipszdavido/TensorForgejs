import * as fs from 'fs';
import * as path from 'path';

export class CSVParser {

    public parseCsvLine(line: string): string[] {
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

    public parse<T = Record<string, string>>(filePath: string): T[] {

        const csvFilePath = path.resolve(filePath);

        if (!fs.existsSync(csvFilePath)) {
            throw new Error(`File not found at target path: ${csvFilePath}`);
        }

        const fileContent = fs.readFileSync(csvFilePath, 'utf-8');
        const lines = fileContent.split(/\r?\n/);

        if (lines.length === 0 || !lines[0]) {
            throw new Error("CSV file is empty");
        }

        const headers = this.parseCsvLine(lines[0]);
        const records: T[] = [];

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const columns = this.parseCsvLine(line);
            const record = {} as Record<string, string>;

            for (let j = 0; j < headers.length; j++) {
                record[headers[j]] = columns[j] !== undefined ? columns[j] : '';
            }

            records.push(record as T);
        }

        return records;
    }
}
