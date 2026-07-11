import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {Tanh} from "../../../math/tanh";
import {Sigmoid} from "../../../math/sigmoid";
import {Pow} from "../../../math/pow";
import {Scalar} from "../../../core/Scalar";

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

        const Wxo_x = Matrix.matrixMulVector(this.Wxo, x)
        const Who_previous_h = Matrix.matrixMulVector(this.Who, previous_h)
        const Wxo_x_Who_previous_h_bo = Wxo_x.add(Who_previous_h).add(this.bo);
        const o_t = Sigmoid(Wxo_x_Who_previous_h_bo);

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

        const tanh_sub = Scalar.one.sub(Pow(Tanh(this.c), 2))
        const dL_dh_o = dL_dh.mulVectors(tanh_sub).mulVectors(this.o)
        const dL_dc = Vector.addVectors(dL_dh_o, gradFromUpLayerC)

        // dL/dc = dL/dh * dh/dc
        // = dL/dh * ot (1 - tanh^2(ct))
        // dL/dh
        // These are coming in.

        // calc weights grads
        // Wxf
        // dL/dWxf = dL/dc * dc/df * df/dzf * dzf/dWxf
        // = dL/dc * cprev * ft(1-ft) * xt
        const dzf = dL_dc.mulVectors(this.previousC).mulVectors(this.f.sub(this.f.mulVectors((this.f))));
        const dL_dWxf = Matrix.outerProduct(dzf, this.x)
        // Whf
        // dL/dWhf = dL/dc * dc/df * df/dzf * dzf/dWhf
        // = dL/dc * cprev * ft(1-ft) * hprev
        const dL_dWhf = Matrix.outerProduct(dzf, this.previousH);


        // Wxi
        // dL/dWxi = dL/dc * dc/di * di/dzi * dzi/dWxi
        // = dL/dc * gt * it(1-it) * xt
        const dzi = dL_dc.mulVectors(this.g).mulVectors(this.i.mulVectors(Scalar.one.sub(this.i)))
        const dL_dWxi = Matrix.outerProduct(dzi, this.x)
        // Whi
        // dL/dWhi = dL/dc * dc/di * di/dzi * dzi/dWhi
        // = dL/dc * gt * it(1-it) * hprev
        const dL_dWhi = Matrix.outerProduct(dzi, this.previousH)


        // Wxo
        // dL/dWxo = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dWxo
        // = dL/dht * tanh(ct) * ot(1-ot) * xt
        const dzo = dL_dh.mulVectors(Tanh(this.c)).mulVectors(this.o.mulVectors(Scalar.one.sub(this.o)))
        const dL_dWxo = Matrix.outerProduct(dzo, this.x)

        // Who
        // dL/dWho = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dwho
        // = dL/dht * tanh(ct) * ot(1-ot) * hprev
        const dL_Who = Matrix.outerProduct(dzo, this.previousH)

        // Wxg
        // dL/dWxg = dL/dc * dc/dg * dg/dzg * dzg/dWxg
        // = dL/dc * it * tanh_derivative(zg) * xt
        const dzg = dL_dc.mulVectors(this.i).mulVectors(Scalar.one.sub(Pow(this.g, 2)))
        const dL_dWxg = Matrix.outerProduct(dzg, this.x)
        // Whg
        // dL/dWhg = dL/dc * dc/dg * dg/dzg * dzg/dWhg
        // = dL/dc * it * tanh_derivative(zg) * hprev
        const dL_dWhg = Matrix.outerProduct(dzg, this.previousH);

        return {
            h: this.h,
            c: this.c,
            dx: Matrix.matrixMulVector(this.Wxf.transpose(), dzf).add(Matrix.matrixMulVector(this.Wxi.transpose(), dzi)).add(Matrix.matrixMulVector(this.Wxg.transpose(), dzg)).add(Matrix.matrixMulVector(this.Wxo.transpose(), dzo)),
            dhPrev: Matrix.matrixMulVector(this.Whf.transpose(), dzf).add(Matrix.matrixMulVector(this.Whi.transpose(), dzi)).add(Matrix.matrixMulVector(this.Whg.transpose(), dzg)).add(Matrix.matrixMulVector(this.Who.transpose(), dzo)),
            dcPrev: dL_dc.mulVectors(Tanh(this.f))
        }

    }

    update(lr: number) {

    }

}
