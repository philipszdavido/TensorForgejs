import {Matrix} from "../../../core/Matrix";
import {Vector} from "../../../core/Vector";
import {NeuralNetworkDense} from "../dense/NeuralNetworkDense";
import {LayerInterface} from "../Types";
import {Convo2D} from "../convo2d/Convo2D";
import {ReLU2D} from "../convo2d/ReLU2DLayer";
import {Flatten} from "../convo2d/Flatten";

export class CNNNetwork implements LayerInterface {

    constructor(
        public readonly conv: Convo2D,
        public readonly relu: ReLU2D,
        public readonly flatten: Flatten,
        public readonly dense: NeuralNetworkDense
    ) {}

    forward(image: Matrix): Vector {

        const c =
            this.conv.forward(image);

        const r =
            this.relu.forward(c);

        const f =
            this.flatten.forward(r);

        return this.dense.forward(
            f.toArray()
        );
    }

    backward(label: number[]) {

        const dFlatten =
            this.dense.backward(label);

        const dRelu =
            this.flatten.backward(dFlatten);

        const dConv =
            this.relu.backward(dRelu);

        this.conv.backward(dConv);
    }

    update(lr: number) {

        this.conv.updateWeights(lr);

        this.dense.update(lr);
    }
}
