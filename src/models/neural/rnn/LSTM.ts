import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {Tanh} from "../../../math/tanh";
import {Sigmoid} from "../../../math/sigmoid";
import {Pow} from "../../../math/pow";
import {Scalar} from "../../../core/Scalar";

export class LSTM {

    constructor(private lstmCells: LSTMCell[]) {
    }

    forward() {

    }

    backward() {
    }

}

export type LayerCell = {
    x: Vector;
    h: Vector;
    // hprev: Vector
    // cprev: Vector;

    f: Vector

    i: Vector;

    g: Vector

    o: Vector;

    c: Vector;
    previousH: Vector;
    previousC: Vector;
};

export class LSTMCell {

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
    private dWxf: Matrix;
    private dWhf: Matrix;
    private dWxi: Matrix;
    private dWhi: Matrix;
    private dWhg: Matrix;
    private dWxo: Matrix;
    private dWho: Matrix;
    private dWxg: Matrix;
    private dbf: Vector;
    private dbi: Vector;
    private dbg: Vector;
    private dbo: Vector;

    private caches: LayerCell[] = []

    constructor(public inputSize: number, public hiddenSize: number) {

        this.x = new Vector(inputSize);
        this.bf = new Vector(hiddenSize);
        this.bf.fill(1);

        this.i = new Vector(hiddenSize);
        this.bi = new Vector(hiddenSize);
        this.g = new Vector(hiddenSize);
        this.bg = new Vector(hiddenSize);
        this.o = new Vector(hiddenSize);
        this.bo = new Vector(hiddenSize);

        this.Wxf = Matrix.random(hiddenSize, inputSize);
        this.Whf = Matrix.random(hiddenSize, hiddenSize);
        this.Wxi = Matrix.random(hiddenSize, inputSize);
        this.Whi = Matrix.random(hiddenSize, hiddenSize);
        this.Wxg = Matrix.random(hiddenSize, inputSize);
        this.Whg = Matrix.random(hiddenSize, hiddenSize);
        this.Wxo = Matrix.random(hiddenSize, inputSize);
        this.Who = Matrix.random(hiddenSize, hiddenSize);

        this.dWxf = Matrix.zeros(hiddenSize, inputSize);
        this.dWhf = Matrix.zeros(hiddenSize, hiddenSize);
        this.dWxi = Matrix.zeros(hiddenSize, inputSize);
        this.dWhi = Matrix.zeros(hiddenSize, hiddenSize);
        this.dWhg = Matrix.zeros(hiddenSize, hiddenSize);
        this.dWxo = Matrix.zeros(hiddenSize, inputSize);
        this.dWho = Matrix.zeros(hiddenSize, hiddenSize);
        this.dWxg = Matrix.zeros(hiddenSize, inputSize);

        this.dbf = new Vector(hiddenSize);
        this.dbi = new Vector(hiddenSize);
        this.dbg = new Vector(hiddenSize);
        this.dbo = new Vector(hiddenSize);

    }

    forward(x: Vector, previous_h: Vector, previous_c: Vector, time: number) {

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

        this.caches[time] = {
            c: c_t,
            f: f_t,
            g: g,
            h: h_t,
            i: i_t,
            o: o_t,
            previousC: previous_c,
            previousH: previous_h,
            x: x

        }

        return {h_t, c_t}

    }

    backward(time: number, gradFromUpLayerH: Vector, gradFromUpLayerC: Vector) {

        const {
            c,
            f,
            g,
            h,
            i,
            o,
            previousC,
            previousH,
            x
        } = this.caches[time]

        const tanhC = Tanh(c);

        const dL_dh = gradFromUpLayerH

        const tanh_sub = Scalar.one.sub(Pow(tanhC, 2))
        const dL_dh_o = dL_dh.mulVectors(tanh_sub).mulVectors(o)
        const dL_dc = Vector.addVectors(dL_dh_o, gradFromUpLayerC)

        // dL/dc = dL/dh * dh/dc
        // = dL/dh * ot (1 - tanh^2(ct))
        // dL/dh
        // These are coming in.

        // calc weights grads
        // Wxf
        // dL/dWxf = dL/dc * dc/df * df/dzf * dzf/dWxf
        // = dL/dc * cprev * ft(1-ft) * xt
        const dzf = dL_dc.mulVectors(previousC).mulVectors(f.sub(f.mulVectors((f))));
        const dL_dWxf = Matrix.outerProduct(dzf, x);
        this.dWxf = this.dWxf.addInPlace(dL_dWxf);
        // Whf
        // dL/dWhf = dL/dc * dc/df * df/dzf * dzf/dWhf
        // = dL/dc * cprev * ft(1-ft) * hprev
        const dL_dWhf = Matrix.outerProduct(dzf, previousH);
        this.dWhf = this.dWhf.addInPlace(dL_dWhf);

        // Wxi
        // dL/dWxi = dL/dc * dc/di * di/dzi * dzi/dWxi
        // = dL/dc * gt * it(1-it) * xt
        const dzi = dL_dc.mulVectors(g).mulVectors(i.mulVectors(Scalar.one.sub(i)))
        const dL_dWxi = Matrix.outerProduct(dzi, x)
        this.dWxi = this.dWxi.addInPlace(dL_dWxi);
        // Whi
        // dL/dWhi = dL/dc * dc/di * di/dzi * dzi/dWhi
        // = dL/dc * gt * it(1-it) * hprev
        const dL_dWhi = Matrix.outerProduct(dzi, previousH)
        this.dWhi = this.dWhi.addInPlace(dL_dWhi)


        // Wxo
        // dL/dWxo = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dWxo
        // = dL/dht * tanh(ct) * ot(1-ot) * xt
        const dzo = dL_dh.mulVectors(tanhC).mulVectors(o.mulVectors(Scalar.one.sub(o)))
        const dL_dWxo = Matrix.outerProduct(dzo, x);
        this.dWxo = this.dWxo.addInPlace(dL_dWxo);

        // Who
        // dL/dWho = dL/dht * dht/dot * dot/dsigmoid * dsigmoid/dwho
        // = dL/dht * tanh(ct) * ot(1-ot) * hprev
        const dL_dWho = Matrix.outerProduct(dzo, previousH)
        this.dWho = this.dWho.addInPlace(dL_dWho)

        // Wxg
        // dL/dWxg = dL/dc * dc/dg * dg/dzg * dzg/dWxg
        // = dL/dc * it * tanh_derivative(zg) * xt
        const dzg = dL_dc.mulVectors(i).mulVectors(Scalar.one.sub(Pow(g, 2)))
        const dL_dWxg = Matrix.outerProduct(dzg, x)
        this.dWxg = this.dWxg.addInPlace(dL_dWxg)
        // Whg
        // dL/dWhg = dL/dc * dc/dg * dg/dzg * dzg/dWhg
        // = dL/dc * it * tanh_derivative(zg) * hprev
        const dL_dWhg = Matrix.outerProduct(dzg, previousH);
        this.dWhg = this.dWhg.addInPlace(dL_dWhg)

        this.dbf = this.dbf.add(dzf);
        this.dbi = this.dbi.add(dzi);
        this.dbg = this.dbg.add(dzg);
        this.dbo = this.dbo.add(dzo);

        return {
            dx: Matrix.matrixMulVector(this.Wxf.transpose(), dzf).add(Matrix.matrixMulVector(this.Wxi.transpose(), dzi)).add(Matrix.matrixMulVector(this.Wxg.transpose(), dzg)).add(Matrix.matrixMulVector(this.Wxo.transpose(), dzo)),
            dhPrev: Matrix.matrixMulVector(this.Whf.transpose(), dzf).add(Matrix.matrixMulVector(this.Whi.transpose(), dzi)).add(Matrix.matrixMulVector(this.Whg.transpose(), dzg)).add(Matrix.matrixMulVector(this.Who.transpose(), dzo)),
            dcPrev: dL_dc.mulVectors(this.f)
        }

    }

    update(lr: number) {

        this.Wxf = this.Wxf.subInPlace(this.dWxf.multiplyScalar(lr));
        this.Whf = this.Whf.subInPlace(this.dWhf.multiplyScalar(lr));
        this.Wxi = this.Wxi.subInPlace(this.dWxi.multiplyScalar(lr));
        this.Whi = this.Whi.subInPlace(this.dWhi.multiplyScalar(lr));
        this.Wxg = this.Wxg.subInPlace(this.dWxg.multiplyScalar(lr));
        this.Whg = this.Whg.subInPlace(this.dWhg.multiplyScalar(lr));
        this.Wxo = this.Wxo.subInPlace(this.dWxo.multiplyScalar(lr));
        this.Who = this.Who.subInPlace(this.dWho.multiplyScalar(lr));

        this.bf = this.bf.sub(this.dbf.mulScalar(lr));
        this.bi = this.bi.sub(this.dbi.mulScalar(lr));
        this.bg = this.bg.sub(this.dbg.mulScalar(lr));
        this.bo = this.bo.sub(this.dbo.mulScalar(lr));

        this.zeroGrad();

    }

    zeroGrad() {

        this.dWxf = this.dWxf.zeros()
        this.dWhf = this.dWhf.zeros()
        this.dWxi = this.dWxi.zeros()
        this.dWhi = this.dWhi.zeros()
        this.dWxo = this.dWxo.zeros()
        this.dWho = this.dWho.zeros()
        this.dWxg = this.dWxg.zeros()
        this.dWhg = this.dWhg.zeros()

        this.dbf = this.dbf.zeros()
        this.dbi = this.dbi.zeros();
        this.dbg = this.dbg.zeros();
        this.dbo = this.dbo.zeros();

    }

}
