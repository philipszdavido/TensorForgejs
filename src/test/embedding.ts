import EmbeddingLayer from "../models/neural/gpt/EmbeddingLayer";

const vocabulary = ["buy", "now", "free", "meeting", "project", "click", "hello"];
const vocabSize = vocabulary.length;

const emde = new EmbeddingLayer(vocabSize, 9);

emde.embeddings.print()

emde.lookup([0, 7]).print()
