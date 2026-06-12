import KMeans, {DataLabel} from "../src/models/KMeans";

let samples: DataLabel[] = [
    {
        data: [150, 1],
        label: "Apple - Red",
    },
    {
        data: [170, 1],
        label: "Apple - Green",
    },
    {
        data: [13, 0],
        label: "Banana - Yellow",
    },
    {
        data: [12, 0],
        label: "Banana - Red",
    },
];

let km = new KMeans(2);

km.train(samples)
console.log(km.getCentroids())
console.log(JSON.stringify(km.getDistances()))
console.log((km.predict(12, 0)))
