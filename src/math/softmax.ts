export default function softmax(logits: number[]) {

    const max = Math.max(...logits);

    const exps = logits.map(
        x => Math.exp(x - max)
    );

    const sum = exps.reduce(
        (a, b) => a + b,
        0
    );

    return exps.map(
        x => x / sum
    );
}
