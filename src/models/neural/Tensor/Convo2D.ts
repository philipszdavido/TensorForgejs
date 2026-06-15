import {NLayer} from "../Types";
import {Tensor} from "../../../core/Tensor";

export class Conv2DLayer implements NLayer {

    private input!: Tensor;

    private kernelGrad!: Tensor;

    constructor(
        private kernel: Tensor,
        private stride = 1,
        private learningRate = 0.01
    ) {
    }

    forward(input: Tensor): Tensor {

        this.input = input;

        const [h, w] = input.shape;
        const [kh, kw] = this.kernel.shape;

        const outH =
            Math.floor((h - kh) / this.stride) + 1;

        const outW =
            Math.floor((w - kw) / this.stride) + 1;

        const output =
            new Tensor([outH, outW]);

        for (let r = 0; r < outH; r++) {
            for (let c = 0; c < outW; c++) {

                let sum = 0;

                for (let kr = 0; kr < kh; kr++) {
                    for (let kc = 0; kc < kw; kc++) {

                        sum +=
                            input.get(
                                r * this.stride + kr,
                                c * this.stride + kc
                            )
                            *
                            this.kernel.get(kr, kc);

                    }
                }

                output.set(
                    sum,
                    r,
                    c
                );
            }
        }

        return output;
    }

    backward(grad: Tensor): Tensor {

        const [h, w] = this.input.shape;
        const [kh, kw] = this.kernel.shape;

        const inputGrad =
            Tensor.zeros([h, w]);

        this.kernelGrad =
            Tensor.zeros([kh, kw]);

        const [outH, outW] = grad.shape;

        // dL/dKernel: the Loss w.r.t to the Kernel
        // How much the kernel contrib. to the Loss
        for (let r = 0; r < outH; r++) {
            for (let c = 0; c < outW; c++) {

                const g = grad.get(r, c);

                for (let kr = 0; kr < kh; kr++) {
                    for (let kc = 0; kc < kw; kc++) {

                        this.kernelGrad.set(
                            this.kernelGrad.get(kr, kc)
                            +
                            g *
                            this.input.get(
                                r * this.stride + kr,
                                c * this.stride + kc
                            ),
                            kr,
                            kc
                        );
                    }
                }
            }
        }

        // dL/dInput
        for (let r = 0; r < outH; r++) {
            for (let c = 0; c < outW; c++) {

                const g = grad.get(r, c);

                for (let kr = 0; kr < kh; kr++) {
                    for (let kc = 0; kc < kw; kc++) {

                        const ir =
                            r * this.stride + kr;

                        const ic =
                            c * this.stride + kc;

                        inputGrad.set(
                            inputGrad.get(ir, ic)
                            +
                            g * this.kernel.get(kr, kc),
                            ir,
                            ic
                        );
                    }
                }
            }
        }

        return inputGrad;
    }

    updateWeights(): void {

        const [kh, kw] = this.kernel.shape;

        for (let r = 0; r < kh; r++) {
            for (let c = 0; c < kw; c++) {

                this.kernel.set(
                    this.kernel.get(r, c)
                    -
                    this.learningRate *
                    this.kernelGrad.get(r, c),
                    r,
                    c
                );
            }
        }
    }

    gradients(): Tensor[] {
        return [this.kernelGrad];
    }

    parameters(): Tensor[] {
        return [];
    }

}
