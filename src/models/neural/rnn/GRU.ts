import {Vector} from "../../../core/Vector";
import {Sigmoid} from "../../../math/sigmoid";
import {Matrix} from "../../../core/Matrix";
import {Tanh} from "../../../math/tanh";
import {Scalar} from "../../../core/Scalar";

export class GRU {

    T: number = 0;

    constructor(
        public gruCell: GRUCell
    ) {
    }

    forward(input_sequence: number[]) {

        this.T = input_sequence.length - 1;

        for (let i = 0; i < input_sequence.length; i++) {
            const x = input_sequence[i];
            const h = this.gruCell.forward(Vector.from([x]), this.gruCell.h_0)
            this.gruCell.h_0 = h;
        }
    }

    backward(outputGrad: Matrix) {
        for (let i = this.T; i >= 0; i--) {
            this.gruCell.backward(outputGrad, i);
        }
    }

}

export class GRUCell {

    h_0: Vector;
    Wz: Matrix;
    Wr: Matrix;
    W: Matrix;
    Uz: Matrix;
    Ur: Matrix;
    U: Matrix;

    constructor(public inputSize: number, public hiddenSize: number) {

        this.h_0 = Vector.zeros(hiddenSize);

        this.Wz = Matrix.random(inputSize, hiddenSize);
        this.Wr = Matrix.random(inputSize, hiddenSize);
        this.W = Matrix.random(inputSize, hiddenSize);

        this.Uz = Matrix.random(hiddenSize, hiddenSize);
        this.Ur = Matrix.random(hiddenSize, hiddenSize);
        this.U = Matrix.random(hiddenSize, hiddenSize);

    }

    forward(x: Vector, h: Vector) {
        const z = Sigmoid(Matrix.matrixMulVector(this.Wz, x).add(Matrix.matrixMulVector(this.Uz, h)))
        const r = Sigmoid(Matrix.matrixMulVector(this.Wr, x).add(Matrix.matrixMulVector(this.Ur, h)))
        const h_prime = Tanh(Matrix.matrixMulVector(this.W, x).add(Matrix.matrixMulVector(this.U, r).mulVectors(h)))

        const h_next = Scalar.one.sub(z).mulVectors(h).add(z.mulVectors(h_prime))

        return h_next;
    }

    backward(outputGrad: Matrix, time: number) {

    }

}
