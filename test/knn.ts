import KNN, {DataLabel} from "../src/models/knn/KNN";

let samples: DataLabel[] = [
    {
        data: [150, 1],
        label: "Apple",
    },
    {
        data: [170, 1],
        label: "Apple",
    },
    {
        data: [130, 0],
        label: "Banana",
    },
    {
        data: [120, 0],
        label: "Banana",
    },
];

const knn = new KNN(samples, 2);

console.log(knn.predict([160, 1]));
