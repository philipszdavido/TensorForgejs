export interface TextFeatures {
    text: string;

    urlCount: number;
    emailCount: number;
    phoneCount: number;
    moneyCount: number;

    uppercaseRatio: number;
    symbolDensity: number;

    repeatedCharScore: number;
}

export class VocabTextCleaner {

    clean(text: string): TextFeatures {

        let t = this.normalizeUnicode(text);

        const urlCount = this.countUrls(t);
        const emailCount = this.countEmails(t);
        const phoneCount = this.countPhones(t);
        const moneyCount = this.countMoney(t);

        const uppercaseRatio = this.computeUppercaseRatio(t);
        const symbolDensity = this.computeSymbolDensity(t);
        const repeatedCharScore = this.computeRepeatedCharScore(t);

        t = this.stripHtml(t);
        t = this.stripQuotedLines(t);
        t = this.replaceUrls(t);
        t = this.replaceEmails(t);
        t = this.replacePhones(t);
        t = this.replaceMoney(t);
        t = this.normalizeSeparators(t);
        t = this.normalizeWhitespace(t);

        return {
            text: t.trim(),

            urlCount,
            emailCount,
            phoneCount,
            moneyCount,

            uppercaseRatio,
            symbolDensity,
            repeatedCharScore
        };
    }

    private normalizeUnicode(text: string): string {
        return text.normalize("NFKC");
    }

    private stripHtml(text: string): string {
        return text.replace(/<[^>]*>/g, " ");
    }

    private stripQuotedLines(text: string): string {
        return text
            .replace(/^>+\s?/gm, " ")
            .replace(/^(from|sent|to|subject):.*$/gim, " ");
    }

    private replaceUrls(text: string): string {
        return text.replace(/https?:\/\/[^\s]+/gi, " <URL> ");
    }

    private replaceEmails(text: string): string {
        return text.replace(
            /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
            " <EMAIL> "
        );
    }

    private replacePhones(text: string): string {
        return text.replace(
            /\b(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b/g,
            " <PHONE> "
        );
    }

    private replaceMoney(text: string): string {
        return text.replace(
            /(\$|€|£|₦)\s?\d+([.,]\d+)?/g,
            " <MONEY> "
        );
    }

    private normalizeSeparators(text: string): string {
        return text.replace(/([=+\-*]{3,}|_{3,}|\.{3,})/g, " <SEP> ");
    }

    private normalizeWhitespace(text: string): string {
        return text.replace(/\s+/g, " ");
    }

    private countUrls(text: string): number {
        return (text.match(/https?:\/\/[^\s]+/gi) || []).length;
    }

    private countEmails(text: string): number {
        return (text.match(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi) || []).length;
    }

    private countPhones(text: string): number {
        return (text.match(/\b(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b/g) || []).length;
    }

    private countMoney(text: string): number {
        return (text.match(/(\$|€|£|₦)\s?\d+([.,]\d+)?/g) || []).length;
    }

    private computeUppercaseRatio(text: string): number {
        const letters = text.replace(/[^a-zA-Z]/g, "");
        if (letters.length === 0) return 0;

        const uppercase = letters.replace(/[^A-Z]/g, "");
        return uppercase.length / letters.length;
    }

    private computeSymbolDensity(text: string): number {
        const total = text.length;
        if (total === 0) return 0;

        const symbols = text.replace(/[a-zA-Z0-9\s]/g, "");
        return symbols.length / total;
    }

    private computeRepeatedCharScore(text: string): number {

        const matches = text.match(/(.)\1{3,}/g);
        if (!matches) return 0;

        let score = 0;

        for (const m of matches) {
            score += m.length;
        }

        return score / text.length;
    }
}
