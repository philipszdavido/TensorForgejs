import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';

const MNIST_DIR = path.join(process.cwd(), 'data', 'mnist');

const FILES = {
    trainImages: 'train-images-idx3-ubyte.gz',
    trainLabels: 'train-labels-idx1-ubyte.gz',
    testImages: 't10k-images-idx3-ubyte.gz',
    testLabels: 't10k-labels-idx1-ubyte.gz'
};

function parseImages(filename: string): number[][] {
    const filePath = path.join(MNIST_DIR, filename);
    if (!fs.existsSync(filePath)) {
        throw new Error(`Missing file at ${filePath}. Make sure your .gz files are in data/mnist/`);
    }

    const compressedBuffer = fs.readFileSync(filePath);
    const buffer = zlib.gunzipSync(compressedBuffer);

    const magicNumber = buffer.readInt32BE(0);
    const numImages = buffer.readInt32BE(4);
    const rows = buffer.readInt32BE(8);
    const cols = buffer.readInt32BE(12);
    const imageSize = rows * cols;

    if (magicNumber !== 2051) throw new Error(`Invalid magic number in ${filename}`);

    console.log(`Extracting ${numImages} local images (${rows}x${cols})...`);

    const images: number[][] = [];
    let offset = 16;

    for (let i = 0; i < numImages; i++) {
        const flatImage = new Array(imageSize);
        for (let p = 0; p < imageSize; p++) {
            flatImage[p] = buffer[offset++] / 255.0;
        }
        images.push(flatImage);
    }
    return images;
}

function parseLabels(filename: string): number[] {
    const filePath = path.join(MNIST_DIR, filename);

    const compressedBuffer = fs.readFileSync(filePath);
    const buffer = zlib.gunzipSync(compressedBuffer);

    const magicNumber = buffer.readInt32BE(0); // Expected: 2049
    const numItems = buffer.readInt32BE(4);

    if (magicNumber !== 2049) throw new Error(`Invalid magic number in ${filename}`);

    console.log(`Extracting ${numItems} local labels...`);

    const labels: number[] = [];
    let offset = 8;

    for (let i = 0; i < numItems; i++) {
        labels.push(buffer[offset++]);
    }
    return labels;
}

export function loadLocalMNIST() {
    return {
        train: {
            images: parseImages(FILES.trainImages),
            labels: parseLabels(FILES.trainLabels)
        },
        test: {
            images: parseImages(FILES.testImages),
            labels: parseLabels(FILES.testLabels)
        }
    };
}

``
