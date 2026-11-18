
// MNIST Dataset Loader
// Adapted from the official TensorFlow.js MNIST example.
// Dataset: 65,000 handwritten digit images.

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const TOTAL_IMAGES = 65000;
const TRAIN_IMAGES = Math.floor(TOTAL_IMAGES * 5 / 6);

const IMAGE_URL =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";

const LABEL_URL =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

class MnistData {

    async load() {

        console.log("Downloading MNIST dataset...");

        // Load handwritten digit images.
        const image = new Image();
        image.crossOrigin = "anonymous";

        const imagePromise = new Promise((resolve, reject) => {

            image.onload = resolve;

            image.onerror = () =>
                reject(new Error("Could not download MNIST images."));

            image.src = IMAGE_URL;
        });

        // Load digit labels.
        const labelPromise = fetch(LABEL_URL);

        const [_, labelResponse] = await Promise.all([
            imagePromise,
            labelPromise
        ]);

        if (!labelResponse.ok) {
            throw new Error("Could not download MNIST labels.");
        }

        const labelBuffer = await labelResponse.arrayBuffer();

        this.datasetLabels = new Uint8Array(labelBuffer);

        if (this.datasetLabels.length !== TOTAL_IMAGES * NUM_CLASSES) {
            throw new Error("Unexpected MNIST label data size.");
        }

        // Convert image sprite into normalized pixel values.
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d", {
            willReadFrequently: true
        });

        const chunkSize = 5000;

        canvas.width = IMAGE_SIZE;
        canvas.height = chunkSize;

        const imageBuffer =
            new Float32Array(TOTAL_IMAGES * IMAGE_SIZE);

        for (let start = 0; start < TOTAL_IMAGES; start += chunkSize) {

            const currentChunk = Math.min(
                chunkSize,
                TOTAL_IMAGES - start
            );

            context.clearRect(0, 0, canvas.width, canvas.height);

            context.drawImage(
                image,
                0, start,
                IMAGE_SIZE, currentChunk,
                0, 0,
                IMAGE_SIZE, currentChunk
            );

            const pixels = context.getImageData(
                0, 0,
                IMAGE_SIZE, currentChunk
            ).data;

            const offset = start * IMAGE_SIZE;

            for (let i = 0; i < currentChunk * IMAGE_SIZE; i++) {
                imageBuffer[offset + i] = pixels[i * 4] / 255;
            }

            console.log(
                `Loaded ${start + currentChunk}/${TOTAL_IMAGES} images`
            );
        }

        this.datasetImages = imageBuffer;

        // Split training and testing data.
        this.trainImages = this.datasetImages.slice(
            0, TRAIN_IMAGES * IMAGE_SIZE
        );

        this.testImages = this.datasetImages.slice(
            TRAIN_IMAGES * IMAGE_SIZE
        );

        this.trainLabels = this.datasetLabels.slice(
            0, TRAIN_IMAGES * NUM_CLASSES
        );

        this.testLabels = this.datasetLabels.slice(
            TRAIN_IMAGES * NUM_CLASSES
        );

        // Create shuffled indices.
        this.trainIndices =
            tf.util.createShuffledIndices(TRAIN_IMAGES);

        this.testIndices =
            tf.util.createShuffledIndices(TOTAL_IMAGES - TRAIN_IMAGES);

        this.trainIndex = 0;
        this.testIndex = 0;

        console.log("MNIST dataset loaded successfully!");
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            this.trainImages,
            this.trainLabels,
            this.trainIndices,
            "train"
        );
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(
            batchSize,
            this.testImages,
            this.testLabels,
            this.testIndices,
            "test"
        );
    }

    nextBatch(batchSize, images, labels, indices, type) {

        const batchImages = new Float32Array(
            batchSize * IMAGE_SIZE
        );

        const batchLabels = new Uint8Array(
            batchSize * NUM_CLASSES
        );

        for (let i = 0; i < batchSize; i++) {

            const indexKey =
                type === "train" ? "trainIndex" : "testIndex";

            const position = this[indexKey] % indices.length;

            const index = indices[position];

            this[indexKey]++;

            batchImages.set(
                images.slice(
                    index * IMAGE_SIZE,
                    (index + 1) * IMAGE_SIZE
                ),
                i * IMAGE_SIZE
            );

            batchLabels.set(
                labels.slice(
                    index * NUM_CLASSES,
                    (index + 1) * NUM_CLASSES
                ),
                i * NUM_CLASSES
            );
        }

        return {
            xs: tf.tensor2d(
                batchImages,
                [batchSize, IMAGE_SIZE]
            ),

            labels: tf.tensor2d(
                batchLabels,
                [batchSize, NUM_CLASSES]
            )
        };
    }

    resetTrainEpoch() {
        this.trainIndices =
            tf.util.createShuffledIndices(
                this.trainImages.length / IMAGE_SIZE
            );

        this.trainIndex = 0;
    }

}
