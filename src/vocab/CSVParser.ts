import * as fs from 'fs';
import * as path from 'path';

export class CSVParser {

    public parse(filePath: string): Record<string, string>[] {
        const csvFilePath = path.resolve(filePath);

        if (!fs.existsSync(csvFilePath)) {
            throw new Error(`File not found at target path: ${csvFilePath}`);
        }

        const fileContent = fs.readFileSync(csvFilePath, 'utf-8');

        const json = this.parseCSV(fileContent);

        return json

    }

    parseCSV(csv: string): Record<string, string>[] {
        const rows: string[][] = [];

        let row: string[] = [];
        let field = "";

        let i = 0;
        let inQuotes = false;

        while (i < csv.length) {
            const char = csv[i];
            const next = csv[i + 1];

            if (inQuotes && char === '"' && next === '"') {
                field += '"';
                i += 2;
                continue;
            }

            if (char === '"') {
                inQuotes = !inQuotes;
                i++;
                continue;
            }

            if (!inQuotes && char === ',') {
                row.push(field);
                field = "";
                i++;
                continue;
            }

            if (!inQuotes && char === '\r' && next === '\n') {
                row.push(field);
                rows.push(row);

                row = [];
                field = "";

                i += 2;
                continue;
            }

            if (!inQuotes && char === '\n') {
                row.push(field);
                rows.push(row);

                row = [];
                field = "";

                i++;
                continue;
            }

            field += char;
            i++;
        }

        row.push(field);
        rows.push(row);

        if (rows.length === 0) {
            return [];
        }

        const headers = rows[0];

        return rows.slice(1).map(row => {
            const obj: Record<string, string> = {};

            headers.forEach((header, index) => {
                obj[header] = row[index] ?? "";
            });

            return obj;
        });
    }

}
