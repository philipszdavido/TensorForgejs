import {Matrix} from "../src/core/Matrix";
import KNN, {DataLabel} from "../src/models/knn/KNN";

const out = new Matrix(3, 2);

out.print();

const A = Matrix.from([
    [1, 2],
    [3, 4],
]);

A.print();
