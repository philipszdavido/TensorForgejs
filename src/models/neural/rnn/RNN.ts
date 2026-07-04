import {Activation, ActivationEnum, ActivationUse, LayerInterface, LossFunction} from "../Types";
import {Vector} from "../../../core/Vector";
import {Matrix} from "../../../core/Matrix";
import transpose from "../../../math/transpose";
import DenseLayer from "../dense/DenseLayer";

// [h(t−1), x(t)]
// │
// ▼
//      Dense0
//         │
//         ▼
//       h(t)   ← save this
//         │
//         ▼
//      Dense1
//         │
//         ▼
//      Dense2
//         │
//         ▼
//       y(t)

export class RNN {

    private outputs: Vector[] = [];
    private T: number = 0;
    private embedding: Embedding;

    constructor(
        public readonly neuralNetwork: NeuralNetworkDenseRNN,
        public readonly rnns: RNNLayer[],
        public readonly hiddenSize: number,
        public embedDim: number,
        public readonly vocabSize: number
    ) {
        this.embedding = new Embedding(vocabSize, embedDim);
    }

    forward(inputSeq: number[]) {

        this.outputs = [];
        this.T = inputSeq.length

        this.clearCache();

        for (let i = 0; i < inputSeq.length; i++) {

            let input = inputSeq[i];
            let inputVec = this.embedding.forward(input).toArray();

            for (let j = 0; j < this.rnns.length; j++) {
                const h = this.rnns[j].forward(inputVec, i == 0 ? this.rnns[j].h0 : this.rnns[j].cache[i - 1].h, i);
                inputVec = h.toArray()
            }

            const out = this.neuralNetwork.forward(inputVec, i);

            this.outputs.push(Vector.from(out.toArray()));

        }

        for (const layer of this.rnns) {
            layer.h0 = Vector.from(
                layer.cache[this.T - 1].h.toArray()
            );
        }

        return this.outputs;

    }

    backward(labels: number[], inputIndices: number[]) {

        let dFuture = this.rnns.map(rnn => Vector.zeros(rnn.hiddenSize));

        for (let t = this.T - 1; t >= 0; t--) {

            let outputGrad = this.neuralNetwork.backward([labels[t]], t);

            for (let j = this.rnns.length - 1; j >= 0; j--) {

                let max = 0;

                for (const row of this.rnns[j].dWhh.toNestedArray())
                    for (const x of row)
                        max = Math.max(max, Math.abs(x));

                let rnnGrad = Vector.addVectors(dFuture[j], outputGrad);

                // send it to the next rnn layer
                const {dx, dPrevH} = this.rnns[j].backward(rnnGrad, t);

                outputGrad = dx;
                dFuture[j] = dPrevH;

            }

            this.embedding.backward(inputIndices[t], outputGrad);
        }

    }

    clearCache() {
        for (const rnn of this.rnns) {
            rnn.cache = [];
        }
    }

    update(lr: number) {
        for (let j = 0; j < this.rnns.length; j++) {
            this.rnns[j].update(lr, this.T);
        }

        this.neuralNetwork.update(lr);
        this.embedding.update(lr);
    }

    resetState() {
        for (const layer of this.rnns) {
            layer.h0 = Vector.zeros(layer.hiddenSize);
        }
        this.neuralNetwork.zeroSoftmax()
    }

    getWeights() {
        return {
            rnns: this.rnns.map(r => r.getWeights()),
            head: this.neuralNetwork.getWeights(),
            embedding: this.embedding.getWeights(),
        };
    }

    // loadWeights(weights: ReturnType<RNN["getWeights"]>) {
    //     this.rnns.forEach((r, i) => r.loadWeights(weights.rnns[i]));
    //     this.neuralNetwork.loadWeights(weights.head);
    // }

    loadWeights(weights: ReturnType<RNN["getWeights"]>) {

        if (weights.rnns.length !== this.rnns.length) {
            throw new Error(
                `Expected ${this.rnns.length} RNN layers but got ${weights.rnns.length}`
            );
        }

        this.rnns.forEach((layer, i) => {
            layer.loadWeights(weights.rnns[i]);
        });

        this.neuralNetwork.setWeights(weights.head);

        this.embedding.setWeights(weights.embedding);

        this.resetState();
    }

}

export type CacheLayer = {
    X: Vector;
    z: Vector;
    h: Vector;
    hPrev: Vector;
};

export class RNNLayer {

    h0: Vector;
    Whh: Matrix

    Wxh: Matrix;

    b: Vector;
    private readonly activation: Activation;

    dWxh: Matrix;
    dWhh: Matrix;

    cache: CacheLayer[] = []
    dB: Vector;

    constructor(public inputSize: number, public hiddenSize: number, public readonly actEnum: ActivationEnum) {

        this.activation = ActivationUse[actEnum];

        this.h0 = Vector.zeros(hiddenSize);
        this.Whh = this.initializeWeights(hiddenSize, hiddenSize, this.activation);

        this.Wxh = this.initializeWeights(hiddenSize, inputSize, this.activation);

        this.b = Vector.zeros(hiddenSize);
        this.dB = Vector.zeros(hiddenSize);

        this.dWxh = Matrix.zeros(hiddenSize, inputSize);
        this.dWhh = Matrix.zeros(hiddenSize, hiddenSize);

    }

    forward(inputSeq: number[], hPrev: Vector = Vector.zeros(this.hiddenSize), time: number) {

        // output = x.Wxh + hPrev.Whh + b
        const X = Vector.from(inputSeq);

        const XWxh = Matrix.matrixMulVector(this.Wxh, X)
        const hPrevWhh = Matrix.matrixMulVector(this.Whh, hPrev)
        const out = Vector.addVectors(XWxh, hPrevWhh)

        const z = Vector.addVectors(out, this.b);

        const h = this.activation.forward(z);

        const cacheLayer = {
            X: X, // input
            z: z, // pre-activation
            h: h, // activation
            hPrev: hPrev, // previous hidden input
        };

        this.cache[time] = cacheLayer;

        return h;

    }

    backward(grad: Vector, time: number) {

        // gradient of Wxh is normal NN
        // gradient of Whh is like normal NN but the input is from  previous time in the series.

        // dZ = dOutput ⊙ activation'(z)
        const dZ = Vector.mulVectors(grad, this.activation.derivative(this.cache[time].h, this.cache[time].h));

        // dWeights = input * dZ
        this.dWxh = Matrix.add(this.dWxh, Matrix.outerProduct(dZ, this.cache[time].X));

        // dWeights = previous input * dZ
        this.dWhh = Matrix.add(this.dWhh, Matrix.outerProduct(dZ, this.cache[time].hPrev));

        this.dB = Vector.addVectors(this.dB, dZ);

        return {
            // dInput = W * dZ
            dx: Matrix.matrixMulVector(transpose(this.Wxh), dZ),
            dPrevH: Matrix.matrixMulVector(transpose(this.Whh), dZ),
        }

    }

    update(lr: number, T: number) {

        const scale = 1 / T;
        this.dWxh = Matrix.multiplyScalar(this.dWxh, scale);
        this.dWhh = Matrix.multiplyScalar(this.dWhh, scale);
        this.dB = Vector.multiplyScalar(this.dB, scale);

        this.Wxh = Matrix.sub(this.Wxh, Matrix.multiplyScalar(this.dWxh, lr))
        this.Whh = Matrix.sub(this.Whh, Matrix.multiplyScalar(this.dWhh, lr))
        this.b = Vector.subVectors(this.b, Vector.multiplyScalar(this.dB, lr))

        // zero out
        this.dB = Vector.zeros(this.hiddenSize);
        this.dWxh = Matrix.zeros(this.hiddenSize, this.inputSize);
        this.dWhh = Matrix.zeros(this.hiddenSize, this.hiddenSize);

    }

    initializeWeightsV1(
        outputs: number,
        inputs: number,
        activation: Activation
    ): Matrix {

        const std = activation.initializer(inputs);

        const W = Matrix.zeros(outputs, inputs);

        for (let r = 0; r < outputs; r++) {
            for (let c = 0; c < inputs; c++) {

                const u1 = Math.random() || 1e-10;
                const u2 = Math.random();
                const norm = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

                W.set(r, c, norm * std);
            }
        }

        return W;
    }

    initializeWeights(outputs: number, inputs: number, activation: Activation): Matrix {
        const scale = Math.sqrt(2 / (inputs + outputs));
        const W = Matrix.zeros(outputs, inputs);

        for (let r = 0; r < outputs; r++) {
            for (let c = 0; c < inputs; c++) {
                const val = (Math.random() * 2 - 1) * scale;
                W.set(r, c, val);
            }
        }
        return W;
    }

    getWeights() {
        return {Wxh: this.Wxh.toNestedArray(), Whh: this.Whh.toNestedArray(), b: this.b.toArray()};
    }

    // loadWeights(weights: { Wxh: number[][]; Whh: number[][]; b: number[] }) {
    //     this.Wxh = Matrix.from(weights.Wxh); //new Matrix(weights.Wxh, this.hiddenSize, this.inputSize);
    //     this.Whh = Matrix.from(weights.Whh); //new Matrix(weights.Whh, this.hiddenSize, this.hiddenSize);
    //     this.b = Vector.from(weights.b);
    // }

    loadWeights(weights: { Wxh: number[][]; Whh: number[][]; b: number[] }) {

        if (
            weights.Wxh.length !== this.hiddenSize ||
            weights.Wxh.some(row => row.length !== this.inputSize)
        ) {
            throw new Error(
                `Invalid Wxh shape. Expected (${this.hiddenSize} x ${this.inputSize})`
            );
        }

        if (
            weights.Whh.length !== this.hiddenSize ||
            weights.Whh.some(row => row.length !== this.hiddenSize)
        ) {
            throw new Error(
                `Invalid Whh shape. Expected (${this.hiddenSize} x ${this.hiddenSize})`
            );
        }

        if (weights.b.length !== this.hiddenSize) {
            throw new Error(
                `Invalid bias size. Expected ${this.hiddenSize}`
            );
        }

        this.Wxh = Matrix.from(weights.Wxh);
        this.Whh = Matrix.from(weights.Whh);
        this.b = Vector.from(weights.b);

        this.dWxh = Matrix.zeros(this.hiddenSize, this.inputSize);
        this.dWhh = Matrix.zeros(this.hiddenSize, this.hiddenSize);
        this.dB = Vector.zeros(this.hiddenSize);

        this.h0 = Vector.zeros(this.hiddenSize);
        this.cache = [];
    }

}

export class SoftmaxCrossEntropy {
    private predictions: Vector[] = [];

    forward(z: Vector, t: number): Vector {
        const raw = z.toArray();
        const maxAttr = Math.max(...raw);

        const exps = raw.map(x => Math.exp(x - maxAttr));
        const sumExps = exps.reduce((a, b) => a + b, 0);

        const softmaxData = exps.map(x => x / sumExps);
        this.predictions[t] = Vector.from(softmaxData);

        return this.predictions[t];
    }

    loss_(y: Vector, pred: Vector): number {
        const yArr = y.toArray();
        const predArr = pred.toArray();
        let totalLoss = 0;

        for (let i = 0; i < yArr.length; i++) {
            totalLoss -= yArr[i] * Math.log(predArr[i] + 1e-15);
        }

        return totalLoss;
    }

    loss(targetIndex: number, pred: Vector) {

        return -Math.log(
            pred.get(targetIndex) + 1e-15
        );

    }

    fusedGradient_(y: Vector, t: number): Vector {

        const predArr = this.predictions[t].toArray();
        const yArr = y.toArray();

        const gradientData = predArr.map((pred, i) => pred - yArr[i]);

        return Vector.from(gradientData);
    }

    fusedGradient(target: number, t: number): Vector {
        const grad = Vector.from(this.predictions[t].toArray());

        grad.set(target, grad.get(target) - 1);

        return grad;
    }

    zero() {
        this.predictions = [];
    }
}

export class NeuralNetworkDenseRNN {
    constructor(
        public readonly denseLayers: DenseLayer[],
        public readonly loss: LossFunction | SoftmaxCrossEntropy,
    ) {
    }

    forward(input: number[], t: number) {

        let a: Vector = Vector.from(input);

        for (let i = 0; i < this.denseLayers.length; i++) {
            const denseLayer = this.denseLayers[i];
            a = denseLayer.forward(a.toArray());
        }

        if (this.loss instanceof SoftmaxCrossEntropy) {
            a = this.loss.forward(a, t);
        }

        return a;

    }

    backward(y: number[], t: number) {

        const output = this.lastDenseLayer();

        const Y: Vector = Vector.from(y);

        let delta: Vector;

        if (this.loss instanceof SoftmaxCrossEntropy) {
            delta = this.loss.fusedGradient(y[0], t);
        } else {

            // get the output error
            const Predicted = output.a!;
            const lossGrad = (this.loss as LossFunction).gradient(Y, Predicted);

            // dZ = dOutput ⊙ activation'(z)
            delta = Vector.mulVectors(lossGrad, output.activation.derivative(output.z!, output.a!));

        }

        for (let i = this.denseLayers.length - 1; i >= 0; i--) {
            const denseLayer = this.denseLayers[i];
            const dA = denseLayer.backward(delta);

            if (i > 0) {

                const prevLayer = this.denseLayers[i - 1]

                delta = Vector.mulVectors(
                    dA,
                    prevLayer.activation.derivative(prevLayer.a!, prevLayer.z!)
                )
            } else delta = dA
        }

        return delta;

    }

    update(lr: number) {
        for (let i = 0; i < this.denseLayers.length; i++) {
            const denseLayer = this.denseLayers[i];
            denseLayer.updateWeights(lr)
        }
    }

    public applyBatchGradients(batchSize: number, learningRate: number) {
        for (const layer of this.denseLayers) {

            layer.dW = Matrix.multiplyScalar(layer.dW, 1 / batchSize);
            layer.dB = Vector.from(layer.dB.toArray().map(b => b / batchSize));

            layer.updateWeights(learningRate);
        }
    }

    lastDenseLayer() {
        return this.denseLayers[this.denseLayers.length - 1];
    }

    public getWeights() {
        return this.denseLayers.map((layer, index) => ({
            layerIndex: index,
            weights: layer.getWeight(),
            biases: layer.getBias()
        }));
    }

    public setWeights(savedLayers: any[]) {
        savedLayers.forEach((savedLayer, index) => {
            this.denseLayers[index].setWeight(savedLayer.weights);
            this.denseLayers[index].setBias(savedLayer.biases);
        });
    }

    zeroSoftmax() {
        if (this.loss instanceof SoftmaxCrossEntropy) {
            this.loss.zero()
        }
    }
}

export class Embedding {
    public weights: Matrix;
    public dWeights: Matrix;

    constructor(public vocabSize: number, public embedDim: number) {
        this.weights = Matrix.multiplyScalar(Matrix.random(vocabSize, embedDim), Math.sqrt(1 / embedDim));
        this.dWeights = Matrix.zeros(vocabSize, embedDim);
    }

    forward(index: number): Vector {
        return Vector.from([...this.weights.getRow(index)]);
    }

    backward(index: number, grad: Vector) {
        for (let i = 0; i < grad.length; i++) {
            this.dWeights.addAt(index, i, grad.get(i));
        }
    }

    update(lr: number) {
        this.weights = Matrix.sub(this.weights, Matrix.multiplyScalar(this.dWeights, lr));
        this.dWeights = Matrix.zeros(this.weights.rows, this.weights.columns);
    }

    getWeights() {
        return this.weights.toNestedArray();
    }

    setWeights(savedLayers: number[][]) {
        this.weights = Matrix.from(savedLayers)
    }
}
