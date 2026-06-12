import NaiveBayes, {DataSet} from "../src/models/NaiveBayes";

const dataset: DataSet[] = [
    {
        data: "click this link to win lottery",
        label: "Spam",
    },
    {
        data: "hello how are you win ?",
        label: "NotSpam",
    },
];

// const nBayes = new NaiveBayes(dataset);
// nBayes.train(["Spam", "NotSpam"]);
// console.log(nBayes.predict("win a lottery"));
//console.log(nBayes._predict("win a lottery"));

const nb = new NaiveBayes(dataset, ["Spam", "NotSpam"]);
nb.train()
console.log(nb.predict("win a lottery"));
