import {Matrix} from "../core/Matrix";
import {Vector} from "../core/Vector";
import {elementwise_addition, elementwise_multiplication, vector_subtract} from "../math/vector/sum";
import {ReLU, ReLU_derivative} from "../math/relu";
import softmax from "../math/softmax";
import transpose from "../math/transpose";

type Layer = {
    weight: Matrix, // W (out × in)
    bias: Vector, // b (out)
    dW: Matrix,
    dB: Vector;
    input?: Vector;
    z?: Vector; // pre-activation
    a?: Vector; // activation
};

export default class NeuralNetwork {

    layers: Layer[] = []

    constructor(public readonly inputSize: number, public readonly hidden: number[], public readonly outputSize: number) {

        let currentInputs = inputSize;

        for (let i = 0; i < hidden.length; i++) {

            const layer = hidden[i]
            // outputs x inputs
            // new Matrix(cols, rows)
            const weight = Matrix.random(layer, currentInputs);
            const dW = Matrix.zeros(layer, currentInputs);
            const bias = Vector.random(layer);
            const dB = Vector.zeros(layer);

            this.layers.push({weight, bias, dW, dB});
            currentInputs = layer;
        }

        const weight = Matrix.random(outputSize, currentInputs);
        const bias = Vector.random(outputSize);
        const dW = Matrix.zeros(outputSize, currentInputs);
        const dB = Vector.zeros(outputSize);

        this.layers.push({weight, bias, dW, dB});

    }

    forward(input: number[]) {

        let a: Vector = Vector.from(input);

        for (let i = 0; i < this.layers.length; i++) {

            a.print("Input")

            const layer = this.layers[i];
            layer.weight.print("Weight")
            layer.bias.print("Bias")

            const z = Matrix.matrixMulVector(layer.weight, a);

            const zWithBias = Vector.from(
                elementwise_addition(z, layer.bias)
            );

            layer.z = zWithBias;
            layer.input = a;

            layer.z.print("zWithBias")

            if (i === this.layers.length - 1) {
                layer.a = Vector.from(softmax(zWithBias.toArray()));
            } else {
                layer.a = ReLU(zWithBias);
            }

            a = layer.a;
            layer.a.print("a")
        }

        return a;
    }

    backward(y: number[]) {

        console.log("==============Backward=================")

        const output = this.layers[this.layers.length - 1];
        const Predicted = output.a!;

        const Y: Vector = Vector.from(y);

        let error = vector_subtract(Predicted, Y);

        Y.print("Y")
        Predicted?.print("Predicted")

        // get the output error

        for (let i = this.layers.length - 1; i >= 0; i--) {

            const layer = this.layers[i];

            error.print("Error")

            layer.weight.print("Weight");
            layer.input?.print("Input " + i)
            layer.z?.print("Z")
            layer.bias?.print("Bias " + i);

            layer.dB = Vector.from(error.toArray());
            layer.dB.print("dB " + i)

            layer.dW = Matrix.outerProduct(error, layer!.input!);

            layer.dW.print("dW " + i);

            if (i > 0) {

                const WT = transpose(layer.weight);

                WT.print("Transpose")
                error.print("Error")

                // error = Vector.vectorMulMatrix(error, layer.weight)
                error = Vector.from(elementwise_multiplication(
                    Matrix.matrixMulVector(
                        WT,
                        error
                    ),
                    ReLU_derivative(this.layers[i - 1].z!)
                ));

                error.print("Error");

            }
        }

    }

    update(lr: number) {

        for (let i = 0; i < this.layers.length; i++) {

            const layer = this.layers[i];

            for (let r = 0; r < layer.weight.rows; r++) {
                for (let c = 0; c < layer.weight.columns; c++) {

                    const current = layer.weight.get(r, c);
                    const grad = layer.dW.get(r, c);
                    console.log("Weight: ", current - lr * grad)
                    layer.weight.set(r, c, current - lr * grad);
                }
            }

            for (let j = 0; j < layer.bias.length; j++) {

                const current = layer.bias.get(j);
                const grad = layer.dB.get(j);
                console.log("Bias:", current - lr * grad)

                layer.bias.set(j, current - lr * grad);
            }
        }
    }

}
