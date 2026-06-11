import {Matrix} from "../core/Matrix";

export default function clip(x: number, limit = 5) {
    return Math.max(-limit, Math.min(limit, x));
}

export function normClip(M: Matrix, maxNorm = 5) {
    let sum = 0;

    for (let r = 0; r < M.rows; r++) {
        for (let c = 0; c < M.columns; c++) {
            const v = M.get(r, c);
            sum += v * v;
        }
    }

    const norm = Math.sqrt(sum);

    if (norm > maxNorm) {
        const scale = maxNorm / norm;

        for (let r = 0; r < M.rows; r++) {
            for (let c = 0; c < M.columns; c++) {
                M.set(r, c, M.get(r, c) * scale);
            }
        }
    }
}
