import {VocabTextCleaner} from "../src/vocab/VocabTokenizer";
import {VocabularyMap} from "../src/vocab/VocabularyMap";
import {EmbeddingLayer} from "../src/vocab/EmbeddingLayer";
import DenseLayer from "../src/models/neural/dense/DenseLayer";
import {ActivationEnum, BCE} from "../src/models";
import {NeuralNetworkDense} from "../src/models/neural/dense/NeuralNetworkDense";
import {SpamModel} from "../src/email/SpamModel";
import {CSVParser} from "../src/vocab/CSVParser";

const parser = new CSVParser();
const Nigerian_Fraud: Record<string, string>[] =
    parser.parse("/Users/chidumennamdi/Downloads/dataset/Phishing_Email_Dataset/Nigerian_Fraud.csv");

const Enron: Record<string, string>[] =
    parser.parse("/Users/chidumennamdi/Downloads/dataset/Phishing_Email_Dataset/Enron.csv");

interface EmailData {
    text: string;
    label: number;
}

const fraudCount = Nigerian_Fraud.length;
const enronCount = Enron.length;

const dataset: EmailData[] = [...Nigerian_Fraud, ...Enron]
    .map(csv => ({
        text: csv["body"] ?? "",
        label: Number(csv["label"])
    }))
    .filter(d => d.text.length > 0 && !Number.isNaN(d.label));

const labelCounts = dataset.reduce(
    (acc, d) => {
        acc[d.label] = (acc[d.label] ?? 0) + 1;
        return acc;
    },
    {} as Record<number, number>
);

console.log(
    `\n=== Dataset Stats ===
Total emails: ${dataset.length}
Nigerian Fraud: ${fraudCount}
Enron: ${enronCount}

Label distribution:
${Object.entries(labelCounts)
        .map(([label, count]) => `  Label ${label}: ${count}`)
        .join("\n")}
`
);

dataset.sort(() => Math.random() - 0.5);

const trainCutoff = Math.floor(dataset.length * 0.8);
const trainData = dataset//.slice(0, trainCutoff);
const testData = dataset//.slice(trainCutoff);

const MAX_VOCAB_SIZE = 5000;
const EMBEDDING_DIM = 64;
const FEATURE_COUNT = 7;

const LEARNING_RATE = 0.001;
const EPOCHS = 5;
const BATCH_SIZE = 8;

const cleaner = new VocabTextCleaner();
const vocab = new VocabularyMap();
const embeddingLayer = new EmbeddingLayer(MAX_VOCAB_SIZE, EMBEDDING_DIM);

const hiddenLayer = new DenseLayer(
    EMBEDDING_DIM + FEATURE_COUNT,
    8,
    ActivationEnum.relu
);

const outputLayer = new DenseLayer(
    8,
    1,
    ActivationEnum.sigmoid
);

const net = new NeuralNetworkDense([hiddenLayer, outputLayer], BCE);

const spamModel = new SpamModel(cleaner, vocab, embeddingLayer, net, EMBEDDING_DIM);

console.log("Building vocabulary...");

for (const email of trainData) {
    const cleaned = cleaner.clean(email.text);
    vocab.addDocuments(cleaned.text);
}

vocab.tokenize(MAX_VOCAB_SIZE);

console.log(`Vocabulary size: ${vocab.size()}`);
console.log("-----------------------------------------");

console.log("Training started...");

for (let epoch = 1; epoch <= EPOCHS; epoch++) {

    let epochLoss = 0;

    trainData.sort(() => Math.random() - 0.5);

    for (let i = 0; i < trainData.length; i += BATCH_SIZE) {

        const batch = trainData.slice(i, i + BATCH_SIZE);

        let batchLoss = 0;

        for (const email of batch) {

            const pred = spamModel.predict(email.text);

            const eps = 1e-7;
            const p = Math.min(Math.max(pred, eps), 1 - eps);

            const loss =
                -(email.label * Math.log(p) +
                    (1 - email.label) * Math.log(1 - p));

            batchLoss += loss;

            spamModel.train(email.text, email.label, LEARNING_RATE);
        }

        epochLoss += batchLoss / batch.length;
    }

    const avgLoss = epochLoss / (trainData.length / BATCH_SIZE);

    console.log(`Epoch ${epoch}/${EPOCHS} - Loss: ${avgLoss.toFixed(6)}`);
}

console.log("-----------------------------------------");
console.log("Training complete.");

console.log("Evaluating model...");

let correct = 0;
let tp = 0, fp = 0, fn = 0;

for (const email of testData) {

    const pred = spamModel.predict(email.text);
    const label = email.label;

    const cls = pred >= 0.5 ? 1 : 0;

    if (cls === label) correct++;

    if (cls === 1 && label === 1) tp++;
    if (cls === 1 && label === 0) fp++;
    if (cls === 0 && label === 1) fn++;

    // console.log(
    //     `[${label}] "${email.text.slice(0, 40)}..." → ${pred.toFixed(4)}`
    // );
}

const accuracy = (correct / testData.length) * 100;
const precision = tp / (tp + fp) || 0;
const recall = tp / (tp + fn) || 0;
const f1 = 2 * ((precision * recall) / (precision + recall)) || 0;

console.log("\n=================================");
console.log("FINAL RESULTS");
console.log("=================================");
console.log(`Accuracy : ${accuracy.toFixed(2)}%`);
console.log(`Precision: ${precision.toFixed(4)}`);
console.log(`Recall   : ${recall.toFixed(4)}`);
console.log(`F1 Score : ${f1.toFixed(4)}`);
console.log("=================================");
