import {Matrix} from "../../../core/Matrix";

export class Flatten {

    forward(input: Matrix) {
        return [...input.toNestedArray().flat(Infinity)];
    }

}
