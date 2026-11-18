

let mnistData;
let model;
let isModelTrained = false;
let lossChart;
let weightChart;
let initialFullWeights = [];

// Store actual ANN weight values
let initialWeightSnapshot = [];
let previousWeightSnapshot = [];

// Five connections from different input neurons
// to the first hidden neuron.
const WEIGHT_INDICES = [0, 128, 256, 384, 512];

const WEIGHT_LABELS = [
    "Pixel 1 → Hidden 1",
    "Pixel 2 → Hidden 1",
    "Pixel 3 → Hidden 1",
    "Pixel 4 → Hidden 1",
    "Pixel 5 → Hidden 1"
];


const EPOCHS = 5;
const BATCH_SIZE = 128;
const STEPS_PER_EPOCH = 150;

async function initializeApp() {

    const status = document.getElementById("status");

    try {

        await tf.ready();

        status.textContent = "TensorFlow.js ready. Loading MNIST...";

        console.log("TensorFlow.js:", tf.version.tfjs);
        console.log("Backend:", tf.getBackend());

        // Load dataset
        mnistData = new MnistData();

        await mnistData.load();

        // Display sample images
        await displaySamples();

       // Build ANN
buildModel();

// Initialize graphs
initializeChart();
initializeWeightChart();

// Display initial weights
displayWeights(0);

        status.textContent =
            "MNIST loaded. ANN is ready for training.";

        status.style.color = "#4ade80";

        document.getElementById("train-btn").disabled = false;
        document.getElementById("train-btn").textContent =
            "Start Training";

        console.log("ANN initialized successfully.");

    } catch (error) {

        status.textContent = "Error: " + error.message;
        status.style.color = "#f87171";

        console.error(error);
    }
}


// ======================================
// DISPLAY MNIST SAMPLE IMAGES
// ======================================

async function displaySamples() {

    const sampleGrid = document.getElementById("sample-grid");

    const examples = mnistData.nextTestBatch(12);

    const labels = examples.labels.argMax(1).dataSync();

    sampleGrid.innerHTML = "";

    for (let i = 0; i < 12; i++) {

        const imageTensor = tf.tidy(() => {

            return examples.xs
                .slice([i, 0], [1, 784])
                .reshape([28, 28, 1]);

        });

        const canvas = document.createElement("canvas");

        canvas.width = 28;
        canvas.height = 28;

        await tf.browser.toPixels(imageTensor, canvas);

        const item = document.createElement("div");

        item.className = "sample-item";

        const label = document.createElement("span");

        label.textContent = "Digit: " + labels[i];

        item.appendChild(canvas);
        item.appendChild(label);

        sampleGrid.appendChild(item);

        imageTensor.dispose();
    }

    examples.xs.dispose();
    examples.labels.dispose();
}


// ======================================
// BUILD NEURAL NETWORK
// ======================================

function buildModel() {

    model = tf.sequential({

        layers: [

            tf.layers.dense({

                inputShape: [784],

                units: 128,

                activation: "relu",

                name: "hidden_layer"

            }),

            tf.layers.dense({

                units: 10,

                activation: "softmax",

                name: "output_layer"

            })

        ]

    });

    model.compile({

        optimizer: tf.train.adam(0.001),

        loss: "categoricalCrossentropy",

        metrics: ["accuracy"]

    });

    model.summary();

    const fullWeights =
    model.layers[0].getWeights()[0];

initialFullWeights =
    Array.from(fullWeights.dataSync());

    // Capture initial weights before training.
initialWeightSnapshot = readSelectedWeights();

previousWeightSnapshot = [...initialWeightSnapshot];

console.log("Initial weights:", initialWeightSnapshot);
}


// ======================================
// INITIALIZE LOSS GRAPH
// ======================================

function initializeChart() {

    const ctx = document
        .getElementById("loss-chart")
        .getContext("2d");

    lossChart = new Chart(ctx, {

        type: "line",

        data: {

            labels: [],

            datasets: [{

                label: "Training Loss",

                data: [],

                borderColor: "#38bdf8",

                backgroundColor: "rgba(56,189,248,0.12)",

                borderWidth: 3,

                fill: true,

                tension: 0.3

            }]

        },

        options: {

            responsive: true,

            animation: false,

            scales: {

                x: {

                    title: {
                        display: true,
                        text: "Epoch",
                        color: "#94a3b8"
                    },

                    ticks: {
                        color: "#cbd5e1"
                    }

                },

                y: {

                    title: {
                        display: true,
                        text: "Loss",
                        color: "#94a3b8"
                    },

                    beginAtZero: true,

                    ticks: {
                        color: "#cbd5e1"
                    }

                }

            },

            plugins: {

                legend: {

                    labels: {
                        color: "#ffffff"
                    }

                }

            }

        }

    });
}


// ======================================
// INITIALIZE WEIGHT HISTORY GRAPH
// ======================================

function initializeWeightChart() {

    const canvas = document.getElementById("weight-chart");

    if (!canvas) {
        console.warn("Weight history chart is disabled.");
        return;
    }

    const ctx = canvas.getContext("2d");

    const colors = [
        "#38bdf8",
        "#a78bfa",
        "#4ade80",
        "#facc15",
        "#fb7185"
    ];

    weightChart = new Chart(ctx, {

        type: "line",

        data: {

            labels: [0],

            datasets: WEIGHT_LABELS.map((label, i) => ({

                label: label,

                data: [initialWeightSnapshot[i]],

                borderColor: colors[i],

                borderWidth: 2,

                tension: 0.3,

                pointRadius: 4

            }))

        },

        options: {

            responsive: true,

            animation: false,

            scales: {

                x: {

                    title: {
                        display: true,
                        text: "Epoch",
                        color: "#94a3b8"
                    },

                    ticks: {
                        color: "#cbd5e1"
                    }

                },

                y: {

                    title: {
                        display: true,
                        text: "Weight Value",
                        color: "#94a3b8"
                    },

                    ticks: {
                        color: "#cbd5e1"
                    }

                }

            },

            plugins: {

                legend: {

                    labels: {
                        color: "#ffffff"
                    }

                }

            }

        }

    });
}



// ======================================
// DISPLAY ACTUAL NETWORK WEIGHTS
// ======================================



// ======================================
// READ ACTUAL ANN WEIGHTS
// ======================================


function readSelectedWeights() {

    const weightTensor =
        model.layers[0].getWeights()[0];

    const values = weightTensor.dataSync();

    return WEIGHT_INDICES.map(index => values[index]);
}


// ======================================
// CALCULATE ACTUAL WEIGHT CHANGES
// ======================================

function calculateWeightChanges() {

    const weightTensor =
        model.layers[0].getWeights()[0];

    const current = weightTensor.dataSync();

    let totalChange = 0;

    for (let i = 0; i < current.length; i++) {

        totalChange += Math.abs(
            current[i] - initialFullWeights[i]
        );

    }

    return totalChange / current.length;
}



// ======================================
// DISPLAY WEIGHT CHANGES
// ======================================

function displayWeights(epoch) {

    const weightContainer =
        document.getElementById("weight-values");

    const currentWeights = readSelectedWeights();

    weightContainer.innerHTML = "";

    currentWeights.forEach((value, i) => {

        const initial = initialWeightSnapshot[i];

        const change = value - initial;

        const item = document.createElement("div");

        item.className = "weight-item";

        item.innerHTML = `
            <span>${WEIGHT_LABELS[i]}</span>

            <strong>${value.toFixed(6)}</strong>

            <small>
                Initial: ${initial.toFixed(6)}
            </small>

            <small>
                Change: ${change >= 0 ? "+" : ""}
                ${change.toFixed(6)}
            </small>
        `;

        weightContainer.appendChild(item);

    });

    // Record the actual weights after this epoch.
    if (epoch > 0) {

        weightChart.data.labels.push(epoch);

        currentWeights.forEach((value, i) => {

            weightChart.data.datasets[i].data.push(value);

        });

        weightChart.update();

    }

    previousWeightSnapshot = [...currentWeights];
}




// ======================================
// TRAIN ANN USING BACKPROPAGATION
// ======================================

async function trainANN() {

    const trainButton =
        document.getElementById("train-btn");

    trainButton.disabled = true;

    trainButton.textContent = "Training...";

    let previousLoss = null;

    try {

        for (let epoch = 0; epoch < EPOCHS; epoch++) {

            mnistData.resetTrainEpoch();

            let totalLoss = 0;
            let totalAccuracy = 0;

            for (
                let step = 0;
                step < STEPS_PER_EPOCH;
                step++
            ) {

                const batch =
                    mnistData.nextTrainBatch(BATCH_SIZE);

                try {

                    // Forward propagation,
                    // loss calculation and backpropagation.
                    const result = await model.trainOnBatch(
                        batch.xs,
                        batch.labels
                    );

                    const metrics = Array.isArray(result)
                        ? result
                        : [result];

                    totalLoss += Number(metrics[0]);

                    totalAccuracy += Number(metrics[1] ?? 0);

                } finally {

                    batch.xs.dispose();
                    batch.labels.dispose();

                }

                // Update progress and keep browser responsive.
                const overallProgress =
                    (
                        (epoch * STEPS_PER_EPOCH + step + 1) /
                        (EPOCHS * STEPS_PER_EPOCH)
                    ) * 100;

                document.getElementById("progress-bar")
                    .style.width = overallProgress + "%";

                document.getElementById("progress-text")
                    .textContent =
                    `Epoch ${epoch + 1}/${EPOCHS} | ` +
                    `Batch ${step + 1}/${STEPS_PER_EPOCH}`;

                if (step % 5 === 0) {

                    await tf.nextFrame();

                }

            }

            // Calculate epoch averages.
            const averageLoss =
                totalLoss / STEPS_PER_EPOCH;

            const averageAccuracy =
                totalAccuracy / STEPS_PER_EPOCH;

            // Update dashboard.
            document.getElementById("epoch")
                .textContent = `${epoch + 1} / ${EPOCHS}`;

            document.getElementById("loss")
                .textContent = averageLoss.toFixed(4);

            document.getElementById("accuracy")
                .textContent =
                (averageAccuracy * 100).toFixed(2) + "%";

            // Add loss to chart.
            lossChart.data.labels.push(epoch + 1);

            lossChart.data.datasets[0].data.push(
                averageLoss
            );

            lossChart.update();

            // Show actual weight changes.
            displayWeights(epoch + 1);

            previousLoss = averageLoss;

            console.log(
                `Epoch ${epoch + 1}:`,
                "Loss:", averageLoss.toFixed(4),
                "Accuracy:", (averageAccuracy * 100).toFixed(2) + "%"
            );

            await tf.nextFrame();

        }

        isModelTrained = true;
        document.getElementById("predict-digit-btn").disabled = false;

        document.getElementById("status").textContent =
            "ANN training completed successfully!";

        document.getElementById("progress-text").textContent =
            "Training completed.";

        document.getElementById("test-status").textContent =
            "Training completed. Ready for testing.";

        await evaluateAndTest();

        trainButton.textContent = "Training Completed";

        console.log("ANN training completed.");

    } catch (error) {

        document.getElementById("status").textContent =
            "Training error: " + error.message;

        document.getElementById("status").style.color =
            "#f87171";

        trainButton.disabled = false;

        trainButton.textContent = "Retry Training";

        console.error(error);

    }

}



// ======================================
// TEST ANN USING UNSEEN MNIST DATA
// ======================================

async function evaluateAndTest() {

    const testStatus =
        document.getElementById("test-status");

    const resultContainer =
        document.getElementById("prediction-result");

    testStatus.textContent =
        "Evaluating ANN using unseen testing images...";

    resultContainer.textContent =
        "Running evaluation...";

    const TEST_BATCH_SIZE = 100;
    const TEST_BATCHES = 100;

    let correct = 0;
    let total = 0;

    let digit9Sample = null;

    try {

        for (let batchNumber = 0;
             batchNumber < TEST_BATCHES;
             batchNumber++) {

            const batch =
                mnistData.nextTestBatch(TEST_BATCH_SIZE);

            const predictionTensor =
                model.predict(batch.xs);

            const predictionValues =
                predictionTensor.dataSync();

            const actualTensor =
                batch.labels.argMax(1);

            const actualValues =
                actualTensor.dataSync();

            for (let i = 0; i < TEST_BATCH_SIZE; i++) {

                const probabilities =
                    predictionValues.slice(i * 10, i * 10 + 10);

                let predictedDigit = 0;

                for (let j = 1; j < 10; j++) {

                    if (probabilities[j] >
                        probabilities[predictedDigit]) {

                        predictedDigit = j;
                    }
                }

                const actualDigit = actualValues[i];

                if (predictedDigit === actualDigit) {
                    correct++;
                }

                total++;

                // Save the first actual digit 9.
                if (
                    actualDigit === 9 &&
                    digit9Sample === null
                ) {

                    digit9Sample = {

                        pixels: batch.xs
                            .slice([i, 0], [1, 784])
                            .dataSync(),

                        probabilities: Array.from(probabilities),

                        predicted: predictedDigit

                    };
                }
            }

            predictionTensor.dispose();
            actualTensor.dispose();

            batch.xs.dispose();
            batch.labels.dispose();

            testStatus.textContent =
                `Testing: ${batchNumber + 1}/${TEST_BATCHES} batches`;

            await tf.nextFrame();
        }

        const accuracy = (correct / total) * 100;

        if (!digit9Sample) {
            throw new Error("No digit 9 found in test samples.");
        }

        testStatus.textContent =
            "Testing completed successfully!";

        displayDigit9(digit9Sample, accuracy, total, correct);

        console.log("Testing samples:", total);
        console.log("Correct predictions:", correct);
        console.log("Test accuracy:", accuracy.toFixed(2) + "%");

    } catch (error) {

        testStatus.textContent =
            "Testing error: " + error.message;

        console.error(error);
    }
}


// ======================================
// DISPLAY DIGIT 9 PREDICTION
// ======================================

function displayDigit9(sample, accuracy, total, correct) {

    const container =
        document.getElementById("prediction-result");

    const canvas = document.createElement("canvas");

    canvas.width = 28;
    canvas.height = 28;

    const context = canvas.getContext("2d");

    const imageData = context.createImageData(28, 28);

    for (let i = 0; i < 784; i++) {

        const pixel = Math.round(sample.pixels[i] * 255);

        imageData.data[i * 4] = pixel;
        imageData.data[i * 4 + 1] = pixel;
        imageData.data[i * 4 + 2] = pixel;
        imageData.data[i * 4 + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);

    const confidence =
        sample.probabilities[sample.predicted] * 100;

    const isCorrect = sample.predicted === 9;

    container.innerHTML = "";

    const heading = document.createElement("h3");

    heading.textContent = "Handwritten Digit Recognition";

    const image = document.createElement("div");

    image.className = "test-image";

    image.appendChild(canvas);

    const actual = document.createElement("p");

    actual.textContent = "Actual digit: 9";

    const predicted = document.createElement("p");

    predicted.textContent =
        "Predicted digit: " + sample.predicted;

    const confidenceText = document.createElement("p");

    confidenceText.textContent =
        "Prediction confidence: " + confidence.toFixed(2) + "%";

    const result = document.createElement("h2");

    result.textContent = isCorrect
        ? "CORRECT PREDICTION!"
        : "INCORRECT PREDICTION";

    result.style.color =
        isCorrect ? "#4ade80" : "#f87171";

    const accuracyText = document.createElement("p");

    accuracyText.textContent =
        `Overall test accuracy: ${accuracy.toFixed(2)}%`;

    const countText = document.createElement("p");

    countText.textContent =
        `Correct: ${correct} / ${total} test images`;

    const probabilityHeading = document.createElement("h3");

    probabilityHeading.textContent =
        "Prediction probabilities (0–9)";

    const probabilityGrid = document.createElement("div");

    probabilityGrid.className = "probability-grid";

    sample.probabilities.forEach((probability, digit) => {

        const item = document.createElement("div");

        item.className = "probability-item";

        item.textContent =
            `${digit}: ${(probability * 100).toFixed(1)}%`;

        probabilityGrid.appendChild(item);
    });

    container.append(
        heading,
        image,
        actual,
        predicted,
        confidenceText,
        result,
        accuracyText,
        countText,
        probabilityHeading,
        probabilityGrid
    );
}



// ======================================
// START TRAINING BUTTON
// ======================================

document.getElementById("train-btn")
    .addEventListener("click", trainANN);


// ======================================
// INITIALIZE APPLICATION
// ======================================

initializeApp();


/* =========================================
   ANN NETWORK VISUALIZATION
========================================= */

(function initializeNetworkVisualization() {

    const container = document.getElementById("network-diagram");
    const status = document.getElementById("network-status");

    const forwardBtn = document.getElementById("forward-btn");
    const backwardBtn = document.getElementById("backward-btn");
    const resetBtn = document.getElementById("reset-network-btn");

    if (!container || !status ||
        !forwardBtn || !backwardBtn || !resetBtn) {
        console.error("Network visualization elements not found.");
        return;
    }

    const SVG_NS = "http://www.w3.org/2000/svg";

    const width = 900;
    const height = 440;

    const inputCount = 5;
    const hiddenCount = 6;
    const outputCount = 10;

    const inputX = 110;
    const hiddenX = 450;
    const outputX = 790;

    const svg = document.createElementNS(SVG_NS, "svg");

    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute(
        "aria-label",
        "Neural network with input, hidden and output layers"
    );

    const edgeLayer = document.createElementNS(SVG_NS, "g");
    const neuronLayer = document.createElementNS(SVG_NS, "g");
    const labelLayer = document.createElementNS(SVG_NS, "g");

    svg.append(edgeLayer, neuronLayer, labelLayer);

    const inputs = [];
    const hidden = [];
    const outputs = [];

    const edges = [];

    function createElement(tag, attributes, parent) {

        const element = document.createElementNS(SVG_NS, tag);

        Object.entries(attributes).forEach(([key, value]) => {
            element.setAttribute(key, value);
        });

        parent.appendChild(element);

        return element;
    }

    function createLabel(x, y, text) {

        const label = createElement("text", {
            x,
            y,
            class: "network-layer-label"
        }, labelLayer);

        label.textContent = text;
    }

    function createNeuron(id, x, y, type, label) {

        const group = createElement("g", {
            id,
            class: "neuron-group"
        }, neuronLayer);

        const circle = createElement("circle", {
            cx: x,
            cy: y,
            r: 15,
            class: `network-neuron ${type}-neuron`,
            id: `${id}-circle`
        }, group);

        const text = createElement("text", {
            x,
            y: y + 4,
            "text-anchor": "middle",
            fill: "#ffffff",
            "font-size": "11",
            "font-weight": "bold",
            "pointer-events": "none"
        }, group);

        text.textContent = label;

        return { id, x, y, circle };
    }

    function connectNeurons(source, target, id) {

        const line = createElement("line", {
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            class: "network-edge",
            id
        }, edgeLayer);

        edges.push({
            line,
            source,
            target
        });
    }

    // Layer labels

    createLabel(inputX, 25, "INPUT LAYER");
    createLabel(hiddenX, 25, "HIDDEN LAYER");
    createLabel(outputX, 25, "OUTPUT LAYER");

    // Create input neurons

    for (let i = 0; i < inputCount; i++) {

        const y = 100 + i * 58;

        inputs.push(
            createNeuron(
                `input-${i}`,
                inputX,
                y,
                "input",
                i + 1
            )
        );
    }

    // Create hidden neurons

    for (let i = 0; i < hiddenCount; i++) {

        const y = 85 + i * 54;

        hidden.push(
            createNeuron(
                `hidden-${i}`,
                hiddenX,
                y,
                "hidden",
                i + 1
            )
        );
    }

    // Create output neurons

    for (let i = 0; i < outputCount; i++) {

        const y = 48 + i * 39;

        outputs.push(
            createNeuron(
                `output-${i}`,
                outputX,
                y,
                "output",
                i
            )
        );
    }

    // Connect input to hidden layer

    inputs.forEach((input, i) => {

        hidden.forEach((neuron, j) => {

            connectNeurons(
                input,
                neuron,
                `input-hidden-${i}-${j}`
            );

        });

    });

    // Connect hidden to output layer

    hidden.forEach((neuron, i) => {

        outputs.forEach((output, j) => {

            connectNeurons(
                neuron,
                output,
                `hidden-output-${i}-${j}`
            );

        });

    });

    // Layer descriptions

    createLabel(inputX, 410, "784 actual inputs");
    createLabel(hiddenX, 410, "128 actual neurons");
    createLabel(outputX, 410, "10 digit classes");

    container.replaceChildren(svg);

    // Animation helpers

    const sleep = ms =>
        new Promise(resolve => setTimeout(resolve, ms));

    function clearAnimation() {

        svg.querySelectorAll(".network-edge").forEach(edge => {
            edge.classList.remove(
                "active-forward",
                "active-backward"
            );
        });

        svg.querySelectorAll(".neuron-group").forEach(neuron => {
            neuron.classList.remove("active-neuron");
        });
    }

    function activateNeurons(neurons) {

        neurons.forEach(neuron => {
            neuron.circle.parentElement.classList.add(
                "active-neuron"
            );
        });
    }

    function activateEdges(sourceLayer, targetLayer, direction) {

        edges.forEach(edge => {

            const belongsToConnection =
                sourceLayer.includes(edge.source) &&
                targetLayer.includes(edge.target);

            if (belongsToConnection) {
                edge.line.classList.add(
                    direction === "forward"
                        ? "active-forward"
                        : "active-backward"
                );
            }

        });
    }

    let animationRunning = false;

    async function forwardPropagation() {

        if (animationRunning) return;

        animationRunning = true;
        clearAnimation();

        forwardBtn.disabled = true;
        backwardBtn.disabled = true;

        status.textContent =
            "Step 1: Forward propagation – Input signals enter the network.";

        activateNeurons(inputs);

        await sleep(900);

        status.textContent =
            "Step 2: Signals travel from input neurons to hidden neurons.";

        activateEdges(inputs, hidden, "forward");
        activateNeurons(hidden);

        await sleep(1200);

        status.textContent =
            "Step 3: Hidden neurons process the weighted inputs.";

        await sleep(900);

        status.textContent =
            "Step 4: Signals reach the output neurons (digits 0–9).";

        activateEdges(hidden, outputs, "forward");
        activateNeurons(outputs);

        await sleep(1200);

        status.textContent =
            "Forward propagation completed. The network has produced an output.";

        forwardBtn.disabled = false;
        backwardBtn.disabled = false;

        animationRunning = false;
    }

    async function backwardPropagation() {

        if (animationRunning) return;

        animationRunning = true;
        clearAnimation();

        forwardBtn.disabled = true;
        backwardBtn.disabled = true;

        status.textContent =
            "Step 1: Calculate the prediction error.";

        activateNeurons(outputs);

        await sleep(900);

        status.textContent =
            "Step 2: Backpropagation sends error gradients towards the hidden layer.";

        activateEdges(hidden, outputs, "backward");
        activateNeurons(hidden);

        await sleep(1200);

        status.textContent =
            "Step 3: Gradients continue towards the input layer.";

        activateEdges(inputs, hidden, "backward");
        activateNeurons(inputs);

        await sleep(1200);

        status.textContent =
            "Backpropagation demonstration completed. Gradients have been visualized.";

        forwardBtn.disabled = false;
        backwardBtn.disabled = false;

        animationRunning = false;
    }

    forwardBtn.addEventListener("click", forwardPropagation);

    backwardBtn.addEventListener("click", backwardPropagation);

    resetBtn.addEventListener("click", () => {

        if (animationRunning) return;

        clearAnimation();

        status.textContent =
            "Network reset. Ready for the next demonstration.";

    });

    console.log("ANN network visualization initialized.");
    console.log("Input neurons displayed:", inputCount);
    console.log("Hidden neurons displayed:", hiddenCount);
    console.log("Output neurons displayed:", outputCount);
    console.log("Total displayed connections:", edges.length);

})();


// =====================================
// INTERACTIVE DIGIT DRAWING
// =====================================

const drawingGrid = document.getElementById("drawing-grid");
const clearDrawingBtn = document.getElementById("clear-drawing-btn");
const predictDigitBtn = document.getElementById("predict-digit-btn");

let isDrawing = false;
let drawnPixels = new Array(784).fill(0);

// Generate 28 x 28 grid
if (drawingGrid) {
    drawingGrid.innerHTML = "";

    for (let i = 0; i < 784; i++) {
        const pixel = document.createElement("div");
        pixel.className = "pixel";
        pixel.dataset.index = i;

        drawingGrid.appendChild(pixel);
    }
}

function activatePixel(pixel) {

    if (!pixel || !pixel.classList.contains("pixel")) return;

    const index = Number(pixel.dataset.index);

    const row = Math.floor(index / 28);
    const col = index % 28;

    // Create a 3 x 3 drawing brush.
    for (let dy = -1; dy <= 1; dy++) {

        for (let dx = -1; dx <= 1; dx++) {

            const newRow = row + dy;
            const newCol = col + dx;

            if (
                newRow >= 0 && newRow < 28 &&
                newCol >= 0 && newCol < 28
            ) {

                const newIndex = newRow * 28 + newCol;

                drawnPixels[newIndex] = 1;

                drawingGrid.children[newIndex]
                    .classList.add("active");
            }
        }
    }
}

// Improved drawing interaction
if (drawingGrid) {

    function drawAtPosition(event) {

        const rect = drawingGrid.getBoundingClientRect();

        const cellWidth = rect.width / 28;
        const cellHeight = rect.height / 28;

        const col = Math.floor(
            (event.clientX - rect.left) / cellWidth
        );

        const row = Math.floor(
            (event.clientY - rect.top) / cellHeight
        );

        if (
            row < 0 || row >= 28 ||
            col < 0 || col >= 28
        ) {
            return;
        }

        const index = row * 28 + col;

        activatePixel(drawingGrid.children[index]);
    }

    drawingGrid.addEventListener("pointerdown", (event) => {

        event.preventDefault();

        isDrawing = true;

        drawingGrid.setPointerCapture(event.pointerId);

        drawAtPosition(event);
    });

    drawingGrid.addEventListener("pointermove", (event) => {

        if (!isDrawing) return;

        event.preventDefault();

        drawAtPosition(event);
    });

    drawingGrid.addEventListener("pointerup", () => {
        isDrawing = false;
    });

    drawingGrid.addEventListener("pointercancel", () => {
        isDrawing = false;
    });

    drawingGrid.addEventListener("lostpointercapture", () => {
        isDrawing = false;
    });
}

// Clear drawing
if (clearDrawingBtn) {
    clearDrawingBtn.addEventListener("click", () => {
        drawnPixels.fill(0);

        drawingGrid.querySelectorAll(".pixel").forEach(pixel => {
            pixel.classList.remove("active");
        });

        document.getElementById("drawing-status").textContent =
            "Draw a digit to get started.";

        document.getElementById("predicted-digit").textContent = "?";
    });
}



function preprocessDrawnPixels() {

    // Create the original 28 x 28 image.
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = 28;
    sourceCanvas.height = 28;

    const sourceCtx = sourceCanvas.getContext("2d");
    const imageData = sourceCtx.createImageData(28, 28);

    // White digit on black background.
    for (let i = 0; i < 784; i++) {

        const value = drawnPixels[i] ? 255 : 0;

        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
    }

    sourceCtx.putImageData(imageData, 0, 0);

    // Find digit boundaries.
    let minX = 28, minY = 28;
    let maxX = -1, maxY = -1;

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {

            if (drawnPixels[y * 28 + x] > 0) {

                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;

    // Resize digit to fit within 20 x 20.
    const scale = 20 / Math.max(boxWidth, boxHeight);

    const newWidth = boxWidth * scale;
    const newHeight = boxHeight * scale;

    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;

    const resizedCtx = resizedCanvas.getContext("2d");

    resizedCtx.fillStyle = "black";
    resizedCtx.fillRect(0, 0, 28, 28);

    resizedCtx.imageSmoothingEnabled = true;

    resizedCtx.drawImage(
        sourceCanvas,
        minX,
        minY,
        boxWidth,
        boxHeight,
        (28 - newWidth) / 2,
        (28 - newHeight) / 2,
        newWidth,
        newHeight
    );

    // Calculate the centre of mass.
    const resizedData = resizedCtx.getImageData(0, 0, 28, 28);

    let totalMass = 0;
    let sumX = 0;
    let sumY = 0;

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {

            const value = resizedData.data[
                (y * 28 + x) * 4
            ];

            totalMass += value;
            sumX += x * value;
            sumY += y * value;
        }
    }

    const centerX = totalMass ? sumX / totalMass : 13.5;
    const centerY = totalMass ? sumY / totalMass : 13.5;

    // Shift centre of mass to image centre.
    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = 28;
    outputCanvas.height = 28;

    const outputCtx = outputCanvas.getContext("2d");

    outputCtx.fillStyle = "black";
    outputCtx.fillRect(0, 0, 28, 28);

    outputCtx.drawImage(
        resizedCanvas,
        13.5 - centerX,
        13.5 - centerY
    );

    // Convert to 784 normalized pixel values.
    const processedImage = outputCtx.getImageData(0, 0, 28, 28);

    const pixels = [];

    for (let i = 0; i < 784; i++) {
        pixels.push(processedImage.data[i * 4] / 255);
    }

    // Update processed image preview.
    const previewCanvas = document.getElementById(
        "processed-preview-canvas"
    );

    if (previewCanvas) {

        const previewCtx = previewCanvas.getContext("2d");

        previewCtx.clearRect(0, 0, 28, 28);
        previewCtx.drawImage(outputCanvas, 0, 0);
    }

    return pixels;
}


// =====================================
// ACTUAL ANN DIGIT PREDICTION
// =====================================

if (predictDigitBtn) {

    predictDigitBtn.addEventListener("click", () => {

        // Check whether ANN has been trained
        if (!isModelTrained) {
            document.getElementById("drawing-status").textContent =
                "Please train the ANN first!";
            return;
        }

        // Check whether user has drawn something
        if (!drawnPixels.some(pixel => pixel > 0)) {
            document.getElementById("drawing-status").textContent =
                "Please draw a digit first!";
            return;
        }

        try {

            
// Preprocess drawing before ANN prediction
const processedPixels = preprocessDrawnPixels();

const probabilities = tf.tidy(() => {

    const input = tf.tensor2d(
        processedPixels,
        [1, 784]
    );

    const prediction = model.predict(input);

    return Array.from(prediction.dataSync());

});


            // Find the digit with highest probability
            let predictedDigit = 0;

            for (let i = 1; i < 10; i++) {

                if (probabilities[i] > probabilities[predictedDigit]) {
                    predictedDigit = i;
                }

            }

            const confidence =
                probabilities[predictedDigit] * 100;

            // Display prediction
            document.getElementById("predicted-digit").textContent =
                predictedDigit;

            document.getElementById("drawing-status").textContent =
                "Prediction completed successfully!";

            const confidenceElement =
                document.getElementById("prediction-confidence");

            if (confidenceElement) {
                confidenceElement.textContent =
                    "Confidence: " + confidence.toFixed(2) + "%";
            }

            console.log("Predicted digit:", predictedDigit);
            console.log("Confidence:", confidence.toFixed(2) + "%");

        } catch (error) {

            console.error("Prediction error:", error);

            document.getElementById("drawing-status").textContent =
                "Prediction error: " + error.message;

        }

    });

}

