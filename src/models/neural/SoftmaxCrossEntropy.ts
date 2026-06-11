import {Vector} from "../../core/Vector";

export class SoftmaxCrossEntropy {
    private lastPredictions?: Vector;

    forward(z: Vector): Vector {
        const raw = z.toArray();
        const maxAttr = Math.max(...raw);

        const exps = raw.map(x => Math.exp(x - maxAttr));
        const sumExps = exps.reduce((a, b) => a + b, 0);

        const softmaxData = exps.map(x => x / sumExps);
        this.lastPredictions = Vector.from(softmaxData);

        return this.lastPredictions;
    }

    loss(y: Vector, pred: Vector): number {
        const yArr = y.toArray();
        const predArr = pred.toArray();
        let totalLoss = 0;

        for (let i = 0; i < yArr.length; i++) {
            totalLoss -= yArr[i] * Math.log(predArr[i] + 1e-15);
        }

        return totalLoss;
    }

    fusedGradient(y: Vector): Vector {
        if (!this.lastPredictions) {
            throw new Error("Forward pass must be called before backward pass.");
        }

        const predArr = this.lastPredictions.toArray();
        const yArr = y.toArray();

        const gradientData = predArr.map((pred, i) => pred - yArr[i]);

        return Vector.from(gradientData);
    }
}
