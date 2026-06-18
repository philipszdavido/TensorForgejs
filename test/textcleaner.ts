import {VocabTextCleaner} from "../src/vocab/VocabTokenizer";
import {VocabularyMap} from "../src/vocab/VocabularyMap";

const v = new VocabTextCleaner()

//const cleaned = v.clean("CLAIM your reward immediately: free crypto here! <p></p>")

const map = new VocabularyMap()
//map.tokenize(cleaned)
map.tokenize(v.clean("Hi Nnamdi, can we review the project dashboard tomorrow?"))
map.tokenize(v.clean("  URGENT!! Claim your FREE crypto reward today.  "))


console.log(map.encode("  URGENT!! Claim your FREE crypto reward today.  "))
console.log(map.getFreq())

console.log(map.getVocab())
