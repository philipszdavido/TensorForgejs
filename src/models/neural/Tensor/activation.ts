type Activation = "relu" | "sigmoid" | "tanh" | "linear";

export function activate(x: number, type: Activation): number {
    switch (type) {
        case "relu": return Math.max(0, x);
        case "sigmoid": return 1 / (1 + Math.exp(-x));
        case "tanh": return Math.tanh(x);
        case "linear": return x;
    }
}

export function activateDerivative(x: number, type: Activation): number {
    switch (type) {
        case "relu": return x > 0 ? 1 : 0;
        case "sigmoid":
            const s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
        case "tanh":
            const t = Math.tanh(x);
            return 1 - t * t;
        case "linear": return 1;
    }
}
