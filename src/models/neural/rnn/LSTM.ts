import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {Tanh} from "../../../math/tanh";
import {Sigmoid} from "../../../math/sigmoid";

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

    g: Vector
    Wxg: Matrix
    Whg: Matrix
    bg: Vector

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
        this.g = new Vector(hiddenSize);
        this.bg = new Vector(hiddenSize);
        this.o = new Vector(hiddenSize);
        this.bo = new Vector(hiddenSize);

        this.Wxf = Matrix.zeros(hiddenSize, inputSize);
        this.Whf = Matrix.zeros(hiddenSize, hiddenSize);
        this.Wxi = Matrix.zeros(hiddenSize, inputSize);
        this.Whi = Matrix.zeros(hiddenSize, hiddenSize);
        this.Wxg = Matrix.zeros(hiddenSize, inputSize);
        this.Whg = Matrix.zeros(hiddenSize, hiddenSize);
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


        const Wxg_x = Matrix.matrixMulVector(this.Wxg, x)
        const Whg_previous_h = Matrix.matrixMulVector(this.Whg, previous_h)
        const g = Tanh(Vector.addVectors(Vector.addVectors(Wxg_x, Whg_previous_h), this.bg))

        const c_t = Vector.addVectors(Vector.mulVectors(f_t, previous_c), Vector.mulVectors(i_t, g))

        const _ = Vector.addVectors(Matrix.matrixMulVector(this.Wxo, x), Matrix.matrixMulVector(this.Who, previous_h))
        const __ = Vector.addVectors(_, this.bo);
        const o_t = Sigmoid(Vector.addVectors(_, __));

        const h_t = Vector.mulVectors(o_t, Tanh(c_t))

        this.f = f_t;
        this.i = i_t;
        this.g = g;
        this.o = o_t;
        this.h = h_t;
        this.c = c_t;
        this.previousH = previous_h;
        this.previousC = previous_c;
        this.x = x;

        return {h_t, c_t}

    }

    backward(gradFromUpLayerH: Vector, gradFromUpLayerC: Vector) {

        const dL_dh = gradFromUpLayerH

        const dL_dc = gradFromUpLayerC.

        // dL/dc = dL/dh * dh/dc
        // = dL/dh * ot (1 - tanh^2(ct))
        // dL/dh
        // These are coming in.

        // calc weights grads
        // Wxf
        // dL/dWxf = dL/dc * dc/df * df/dzf * dzf/dWxf
        // = dL/dc * cprev * ft(1-ft) * xt
        // Whf
        // dL/dWhf = dL/dc * dc/df * df/dzf * dzf/dWhf
        // = dL/dc * cprev * ft(1-ft) * hprev


        // Wxi
        // dL/dWxi = dL/dc * dc/di * di/dzi * dzi/dWxi
        // = dL/dc * gt * it(1-it) * xt
        // Whi
        // dL/dWhi = dL/dc * dc/di * di/dzi * dzi/dWhi
        // = dL/dc * gt * it(1-it) * hprev


        // Wxo
        // dL/dWxo = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dWxo
        // = dL/dht * tanh(ct) * ot(1-ot) * xt

        // Who
        // dL/dWho = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dwho
        // = dL/dht * tanh(ct) * ot(1-ot) * hprev

        // Wxg
        // dL/dWxg = dL/dc * dc/dg * dg/dzg * dzg/dWxg
        // = dL/dc * it * tanh_derivative(zg) * xt
        // Whg
        // dL/dWhg = dL/dc * dc/dg * dg/dzg * dzg/dWhg
        // = dL/dc * it * tanh_derivative(zg) * hprev

    }

    update(lr: number) {

    }

}
