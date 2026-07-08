import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {activate} from "../Tensor/activation";
import {Tanh} from "../../../math/tanh";
import {Sigmoid} from "../../../math/sigmoid";
import {elementwise_multiplication} from "../../../math/vector/sum";

export class LSTM {

    x: Vector;
    h!: Vector;
    // hprev: Vector
    // cprev: Vector;

    f!: Vector
    bf: Vector
    Wxf: Matrix
    Whf: Matrix

    i: Vector;
    Wxi: Matrix
    Whi: Matrix
    bi: Vector

    c_tilde: Vector
    Wxc_tilde: Matrix
    Whc_tilde: Matrix
    bc_tilde: Vector

    o: Vector;
    Wxo: Matrix
    Who: Matrix
    bo: Vector

    c!: Vector;
    previousH!: Vector;
    previousC!: Vector;

    constructor(public inputSize: number, public hiddenSize: number) {

        this.x = new Vector(inputSize);
        this.bf = new Vector(hiddenSize);
        this.i = new Vector(hiddenSize);
        this.bi = new Vector(hiddenSize);
        this.c_tilde = new Vector(hiddenSize);
        this.bc_tilde = new Vector(hiddenSize);
        this.o = new Vector(hiddenSize);
        this.bo = new Vector(hiddenSize);

        this.Wxf = Matrix.zeros(hiddenSize, inputSize);
        this.Whf = Matrix.zeros(hiddenSize, hiddenSize);
        this.Wxi = Matrix.zeros(hiddenSize, inputSize);
        this.Whi = Matrix.zeros(hiddenSize, hiddenSize);
        this.Wxc_tilde = Matrix.zeros(hiddenSize, inputSize);
        this.Whc_tilde = Matrix.zeros(hiddenSize, hiddenSize);
        this.Wxo = Matrix.zeros(hiddenSize, inputSize);
        this.Who = Matrix.zeros(hiddenSize, hiddenSize);

    }

    forward(x: Vector, previous_h: Vector, previous_c: Vector) {

        const Wxf_x = Matrix.matrixMulVector(this.Wxf, x)
        const Whf_previous_h = Matrix.matrixMulVector(this.Whf, previous_h)
        const f_t = Sigmoid(Vector.addVectors((Vector.addVectors(Wxf_x, Whf_previous_h)), this.bf));

        const Wxi_x = Matrix.matrixMulVector(this.Wxi, x);
        const Whi_previous_h = Matrix.matrixMulVector(this.Whi, previous_h);
        const i_t = Sigmoid(Vector.addVectors((Vector.addVectors(Wxi_x, Whi_previous_h)), this.bi));


        const Wxc_tilde_x = Matrix.matrixMulVector(this.Wxc_tilde, x)
        const Whc_tilde_previous_h = Matrix.matrixMulVector(this.Whc_tilde, previous_h)
        const c_tilde = Tanh(Vector.addVectors(Vector.addVectors(Wxc_tilde_x, Whc_tilde_previous_h), this.bc_tilde))

        const c_t = Vector.addVectors(Vector.mulVectors(f_t, previous_c), Vector.mulVectors(i_t, c_tilde))

        const _ = Vector.addVectors(Matrix.matrixMulVector(this.Wxo, x), Matrix.matrixMulVector(this.Who, previous_h))
        const __ = Vector.addVectors(_, this.bo);
        const o_t = Sigmoid(Vector.addVectors(_, __));

        const h_t = Vector.mulVectors(o_t, Tanh(c_t))

        this.f = f_t;
        this.i = i_t;
        this.c_tilde = c_tilde;
        this.o = o_t;
        this.h = h_t;
        this.c = c_t;
        this.previousH = previous_h;
        this.previousC = previous_c;
        this.x = x;

        return {h_t, c_t}

    }

    backward() {

    }

    update(lr: number) {

    }

}
