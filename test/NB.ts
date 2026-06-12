// export default class NaiveBayes {
//   private model: Record<string, Record<string, number>> = {};
//   private priorProbabilities: Record<string, number> = {};
//   private sums: Record<string, number> = {};
//   private labels: string[] = [];

//   private totalWordsPerLabel: Record<string, number> = {}; //
//   private vocabulary: Set<string> = new Set(); //

//   constructor(public readonly sample: DataSet[]) {}

//   __predict(text: string) {
//     const total = this.sample.length;

//     const labelProbs: Record<string, number[]> = {};

//     for (const label in this.priorProbabilities) {
//       labelProbs[label] = [];
//     }

//     const words = this.tokenize(text);
//     for (const word of words) {
//       if (this.model[word]) {
//         for (const label in this.labels) {
//           labelProbs[label].push(
//             this.model[word][label] /*this.sums[label] * total*/ /
//               (this.priorProbabilities[label] * total * total),
//           );
//         }
//       }
//     }

//     const prods: Record<string, number> = {};
//     let sum = 0;

//     for (const label in labelProbs) {
//       const result = Math.log(
//         Vector.from(labelProbs[label]).mul() * this.sums[label],
//       );
//       prods[label] = result;
//       sum += result;
//     }

//     console.log(prods);
//     const prob: Record<string, number> = {};
//     for (const label in prods) {
//       prob[label] = prods[label] / sum;
//     }

//     // console.log(prob, prods, labelProbs, this.sums);
//     return prob;
//   }

//   _predict(text: string) {
//     const total = this.sample.length;

//     const labelProbs: Record<string, number[]> = {};

//     for (const label in this.priorProbabilities) {
//       labelProbs[label] = [];
//     }

//     const words = this.tokenize(text);
//     for (const word of words) {
//       if (this.model[word]) {
//         for (const label of this.labels) {
//           labelProbs[label].push(
//             this.model[word][label] /
//               (this.priorProbabilities[label] * total * total),
//           );
//         }
//       }
//     }

//     const prods: Record<string, number> = {};
//     let sum = 0;

//     for (const label in labelProbs) {
//       const productResult =
//         labelProbs[label].reduce((a, b) => a * b, 1) * this.sums[label];
//       prods[label] = productResult;
//       sum += productResult;
//     }

//     console.log(prods);
//     const prob: Record<string, number> = {};
//     for (const label in prods) {
//       prob[label] = sum === 0 ? 0 : prods[label] / sum;
//     }

//     // console.log(prob, prods, labelProbs, this.sums);
//     return prob;
//   }

//   predict(text: string): string {
//     const words = this.tokenize(text);
//     const scores: Record<string, number> = {};
//     const vocabSize = this.vocabulary.size;

//     console.log(this.priorProbabilities);

//     for (const label in this.priorProbabilities) {
//       if (!Object.hasOwn(this.priorProbabilities, label)) continue;

//       scores[label] = Math.log(this.priorProbabilities[label]);

//       for (const word of words) {
//         const wordCountInLabel = this.model[word]
//           ? this.model[word][label] || 0
//           : 0;
//         const totalWordsInLabel = this.totalWordsPerLabel[label] || 0;

//         const wordProbability =
//           (wordCountInLabel + 1) / (totalWordsInLabel + vocabSize);

//         console.log(
//           word,
//           label,
//           wordCountInLabel,
//           totalWordsInLabel,
//           vocabSize,
//           wordProbability,
//         );

//         scores[label] += Math.log(wordProbability);
//       }
//     }

//     console.log(this.totalWordsPerLabel, this.model, this.sums, scores);
//     console.log(scores);
//     return Object.keys(scores).reduce((a, b) =>
//       scores[a] > scores[b] ? a : b,
//     );
//   }

//   train(labels: string[]) {
//     this.labels = labels;

//     const sums = this.findSums();
//     this.sums = sums;
//     const priorProbabilities = this.findPriorProbabilities(sums);

//     for (const label of labels) {
//       this.totalWordsPerLabel[label] = 0;
//     }

//     this.priorProbabilities = priorProbabilities;
//     this.model = this.findPosteriors(labels);
//   }

//   // model = { lottery: { spam: 90, ... }, ... }
//   findPosteriors(labels: string[]) {
//     const model: Record<string, Record<string, number>> = {};

//     for (let i = 0; i < this.sample.length; i++) {
//       const el = this.sample[i];
//       const words = this.tokenize(el.data);

//       for (let j = 0; j < words.length; j++) {
//         const word = words[j];
//         this.vocabulary.add(word); //

//         if (!model[word]) {
//           model[word] = this.initModel(labels);
//         }

//         model[word][el.label] = (model[word][el.label] || 0) + 1;
//         this.totalWordsPerLabel[el.label] += 1; //
//       }
//     }
//     return model;
//   }

//   // { spam: 90/1000 -> 0.9, ... }
//   findPriorProbabilities(sums: Record<string, number>) {
//     const priorProbabilities: Record<string, number> = {};

//     for (const key in sums) {
//       if (!Object.hasOwn(sums, key)) continue;

//       const sum = sums[key];
//       priorProbabilities[key] = sum / this.sample.length;
//     }

//     return priorProbabilities;
//   }

//   // { spam: 90 (num of emails that are spam), non: 90 }
//   findSums() {
//     const sums: Record<string, number> = {};

//     for (let index = 0; index < this.sample.length; index++) {
//       const label = this.sample[index].label;
//       sums[label] = (sums[label] == undefined ? 0 : sums[label]) + 1;
//     }

//     return sums;
//   }

//   initModel(labels: string[]) {
//     const t: Record<string, number> = {};
//     for (let index = 0; index < labels.length; index++) {
//       const label = labels[index];
//       t[label] = 0;
//     }
//     return t;
//   }

//   tokenize(text: string) {
//     const chars = [".", ",", "[", "]", "(", ")"];
//     return text.split(" ");
//   }
// }
