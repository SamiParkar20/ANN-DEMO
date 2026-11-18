

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

            // Input Layer: 784 neurons
            tf.layers.dense({
                inputShape: [784],
                units: 128,
                activation: "relu",
                name: "hidden_layer_1"
            }),

            // Second Hidden Layer: 64 neurons
            tf.layers.dense({
                units: 64,
                activation: "relu",
                name: "hidden_layer_2"
            }),

            // Output Layer: 10 neurons
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

    // Capture initial weights from the first hidden layer
    const fullWeights =
        model.layers[0].getWeights()[0];

    initialFullWeights =
        Array.from(fullWeights.dataSync());

    initialWeightSnapshot = readSelectedWeights();

    previousWeightSnapshot = [...initialWeightSnapshot];

    console.log("Updated ANN Architecture:");
    console.log("784 → 128 → 64 → 10");

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

    heading.textContent =
    "MNIST Test Dataset – Random Example";
    const explanation = document.createElement("p");

explanation.textContent =
    "This is a randomly selected handwritten digit " +
    "from the MNIST testing dataset (10,000 unseen images). " +
    "It is independent of the digit drawn above.";

explanation.style.color = "#94a3b8";
explanation.style.textAlign = "center";
explanation.style.fontSize = "15px";
explanation.style.margin = "15px auto 25px";
explanation.style.maxWidth = "700px";

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

    
// ======================================
// DISPLAY ALL 10 PREDICTION PROBABILITIES
// ======================================

const probabilityHeading = document.createElement("h3");

probabilityHeading.textContent =
    "Prediction Probabilities (0–9)";

const probabilitySubtitle = document.createElement("p");

probabilitySubtitle.className = "probability-subtitle";

probabilitySubtitle.textContent =
    "The model assigns a probability to each possible digit.";

const probabilityGrid = document.createElement("div");

probabilityGrid.className = "probability-list";

sample.probabilities.forEach((probability, digit) => {

    const percentage = probability * 100;

    const isPredicted = digit === sample.predicted;

    // Individual probability row
    const item = document.createElement("div");

    item.className = isPredicted
        ? "probability-row is-predicted"
        : "probability-row";

    // Digit label
    const digitLabel = document.createElement("span");

    digitLabel.className = "probability-digit";

    digitLabel.textContent = digit;

    // Progress bar background
    const track = document.createElement("div");

    track.className = "probability-track";

    // Actual probability bar
    const fill = document.createElement("div");

    fill.className = isPredicted
        ? "probability-fill is-predicted-fill"
        : "probability-fill";

    fill.style.width = percentage + "%";

    track.appendChild(fill);

    // Probability percentage
    const value = document.createElement("span");

    value.className = "probability-value";

    value.textContent = percentage.toFixed(2) + "%";

    // Predicted indicator
    if (isPredicted) {

        const badge = document.createElement("span");

        badge.className = "predicted-badge";

        badge.textContent = "PREDICTED";

        item.append(
            digitLabel,
            track,
            value,
            badge
        );

    } else {

        item.append(
            digitLabel,
            track,
            value
        );

    }

    probabilityGrid.appendChild(item);

});

// Explanation below probability bars
const probabilityExplanation = document.createElement("div");

probabilityExplanation.className = "probability-explanation";

probabilityExplanation.innerHTML = `
    <strong>Why was digit ${sample.predicted} selected?</strong>
    <p>
        The output neuron for digit ${sample.predicted}
        produced the highest probability among all
        ten possible classes.
    </p>
    <small>
        The model selects the class with the highest output score.
    </small>
`;


    
container.append(
    heading,
    explanation,
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
   FOUR-LAYER ANN LEARNING VISUALIZATION
========================================= */

(function initializeNetworkVisualization() {

    const container = document.getElementById("network-diagram");
    const status = document.getElementById("network-status");

    const forwardBtn = document.getElementById("forward-btn");
    const backwardBtn = document.getElementById("backward-btn");
    const resetBtn = document.getElementById("reset-network-btn");

    if (!container || !status ||
        !forwardBtn || !backwardBtn || !resetBtn) return;

    const NS = "http://www.w3.org/2000/svg";

    const W = 1100;
    const H = 500;

    // Four layer positions
    const inputX = 110;
    const hidden1X = 390;
    const hidden2X = 710;
    const outputX = 990;

    // Display representative neurons
    const inputCount = 8;
    const hidden1Count = 8;
    const hidden2Count = 8;
    const outputCount = 10;

    let running = false;

    let svg;

    let edges = [];

    let inputNodes = [];
    let hidden1Nodes = [];
    let hidden2Nodes = [];
    let outputNodes = [];

    function element(tag, attrs, parent) {

        const el = document.createElementNS(NS, tag);

        Object.entries(attrs).forEach(([key, value]) => {
            el.setAttribute(key, value);
        });

        parent.appendChild(el);

        return el;
    }

    function label(x, y, text, size = 17) {

        const el = element("text", {
            x,
            y,
            fill: "#e2e8f0",
            "font-size": size,
            "font-weight": "bold",
            "text-anchor": "middle"
        }, svg);

        el.textContent = text;

        return el;
    }

    function createNeuron(x, y, type, text) {

        const group = element("g", {
            class: "neuron-group"
        }, svg);

        const circle = element("circle", {
            cx: x,
            cy: y,
            r: 20,
            class: `network-neuron ${type}-neuron`
        }, group);

        const value = element("text", {
            x,
            y: y + 4,
            fill: "#ffffff",
            "font-size": "10",
            "font-weight": "bold",
            "text-anchor": "middle",
            "pointer-events": "none"
        }, group);

        value.textContent = text;

        return {
            group,
            circle,
            value,
            x,
            y,
            type
        };
    }

    function connect(source, target, layer) {

        const line = element("line", {
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            class: "network-edge"
        }, svg);

        line.setAttribute("data-layer", layer);

        edges.push({
            line,
            layer
        });

    }

    function buildNetwork() {

        container.innerHTML = "";

        edges = [];

        inputNodes = [];
        hidden1Nodes = [];
        hidden2Nodes = [];
        outputNodes = [];

        svg = element("svg", {
            viewBox: `0 0 ${W} ${H}`,
            role: "img",
            "aria-label": "Four-layer artificial neural network"
        }, container);

        // Layer headings

        label(inputX, 35, "INPUT LAYER");
        label(hidden1X, 35, "HIDDEN LAYER 1");
        label(hidden2X, 35, "HIDDEN LAYER 2");
        label(outputX, 35, "OUTPUT LAYER");

        label(inputX, 65, "784 neurons", 12);
        label(hidden1X, 65, "128 neurons", 12);
        label(hidden2X, 65, "64 neurons", 12);
        label(outputX, 65, "10 classes", 12);

        // INPUT LAYER

        for (let i = 0; i < inputCount; i++) {

            const y = 105 + i * 43;

            const node = createNeuron(
                inputX,
                y,
                "input",
                "0.00"
            );

            inputNodes.push(node);
        }

        // HIDDEN LAYER 1

        for (let i = 0; i < hidden1Count; i++) {

            const y = 105 + i * 43;

            const node = createNeuron(
                hidden1X,
                y,
                "hidden",
                "0.00"
            );

            hidden1Nodes.push(node);
        }

        // HIDDEN LAYER 2

        for (let i = 0; i < hidden2Count; i++) {

            const y = 105 + i * 43;

            const node = createNeuron(
                hidden2X,
                y,
                "hidden",
                "0.00"
            );

            hidden2Nodes.push(node);
        }

        // OUTPUT LAYER

        for (let i = 0; i < outputCount; i++) {

            const y = 85 + i * 36;

            const node = createNeuron(
                outputX,
                y,
                "output",
                String(i)
            );

            outputNodes.push(node);
        }

        
// INPUT -> HIDDEN 1
// Show only representative connections

inputNodes.forEach((input, i) => {

    for (let k = 0; k < 3; k++) {

        const targetIndex = (i * 2 + k * 3) % hidden1Nodes.length;

        connect(
            input,
            hidden1Nodes[targetIndex],
            "forward-1"
        );

    }

});


// HIDDEN 1 -> HIDDEN 2

hidden1Nodes.forEach((hidden, i) => {

    for (let k = 0; k < 3; k++) {

        const targetIndex = (i * 2 + k * 3) % hidden2Nodes.length;

        connect(
            hidden,
            hidden2Nodes[targetIndex],
            "forward-2"
        );

    }

});


// HIDDEN 2 -> OUTPUT

hidden2Nodes.forEach((hidden, i) => {

    for (let k = 0; k < 3; k++) {

        const targetIndex = (i * 3 + k * 2) % outputNodes.length;

        connect(
            hidden,
            outputNodes[targetIndex],
            "forward-3"
        );

    }

});


        label(250, 475, "WEIGHTED CONNECTIONS", 12);
        label(550, 475, "FEATURE EXTRACTION", 12);
        label(850, 475, "CLASSIFICATION", 12);
        // Keep all neuron circles and values above connection lines
[
    ...inputNodes,
    ...hidden1Nodes,
    ...hidden2Nodes,
    ...outputNodes
].forEach(node => {
    svg.appendChild(node.group);
});

    }

    function clearAnimation() {

        edges.forEach(edge => {

            edge.line.classList.remove(
                "active-forward",
                "active-backward"
            );

        });

        [
            ...inputNodes,
            ...hidden1Nodes,
            ...hidden2Nodes,
            ...outputNodes
        ].forEach(node => {

            node.group.classList.remove("active-neuron");

        });

    }

    function activateNodes(nodes) {

        nodes.forEach(node => {

            node.group.classList.add("active-neuron");

        });

    }

    
async function activateConnections(layer, direction) {

    const matchingEdges = edges.filter(
        edge => edge.layer === layer
    );

    const activeClass = direction === "forward"
        ? "active-forward"
        : "active-backward";

    // Animate 3 connections at a time
    for (let i = 0; i < matchingEdges.length; i += 3) {

        const group = matchingEdges.slice(i, i + 3);

        group.forEach(edge => {
            edge.line.classList.add(activeClass);
        });

        await sleep(250);

        group.forEach(edge => {
            edge.line.classList.remove(activeClass);
        });

    }

}


    function sleep(ms) {

        return new Promise(resolve => setTimeout(resolve, ms));

    }

    function updateNodeValues(nodes, values) {

        nodes.forEach((node, i) => {

            node.value.textContent =
                Number(values[i] || 0).toFixed(2);

        });

    }

    // Get actual input from drawing or MNIST

    async function getInputSample() {

        if (typeof drawnPixels !== "undefined" &&
            drawnPixels.some(pixel => pixel > 0)) {

            return {
                values: preprocessDrawnPixels(),
                source: "your drawing"
            };

        }

        const batch = mnistData.nextTestBatch(1);

        const values = Array.from(batch.xs.dataSync());

        batch.xs.dispose();
        batch.labels.dispose();

        return {
            values,
            source: "a sample from MNIST testing data"
        };

    }

    // ======================================
    // FORWARD PROPAGATION
    // ======================================

    async function forwardPropagation() {

        if (running) return;

        if (!isModelTrained || !model || !mnistData) {

            status.textContent =
                "Please train the ANN first before visualizing.";

            return;
        }

        running = true;

        forwardBtn.disabled = true;
        backwardBtn.disabled = true;
        resetBtn.disabled = true;

        clearAnimation();

        let inputTensor;
        let hidden1Tensor;
        let hidden2Tensor;
        let outputTensor;

        try {

            const sample = await getInputSample();

            inputTensor = tf.tensor2d(
                sample.values,
                [1, 784]
            );

            // Actual first hidden layer activation

            hidden1Tensor = model.layers[0].apply(inputTensor);

            const hidden1Values =
                Array.from(hidden1Tensor.dataSync());

            // Actual second hidden layer activation

            hidden2Tensor = model.layers[1].apply(hidden1Tensor);

            const hidden2Values =
                Array.from(hidden2Tensor.dataSync());

            // Actual output probabilities

            outputTensor = model.predict(inputTensor);

            const probabilities =
                Array.from(outputTensor.dataSync());

            // Select strongest 8 input pixels

            const selectedInputs = sample.values
                .map((value, index) => ({ value, index }))
                .sort((a, b) => b.value - a.value)
                .slice(0, 8);

            // Select strongest 8 first hidden neurons

            const selectedHidden1 = hidden1Values
                .map((value, index) => ({ value, index }))
                .sort((a, b) => b.value - a.value)
                .slice(0, 8);

            // Select strongest 8 second hidden neurons

            const selectedHidden2 = hidden2Values
                .map((value, index) => ({ value, index }))
                .sort((a, b) => b.value - a.value)
                .slice(0, 8);

            updateNodeValues(
                inputNodes,
                selectedInputs.map(item => item.value)
            );

            updateNodeValues(
                hidden1Nodes,
                selectedHidden1.map(item => item.value)
            );

            updateNodeValues(
                hidden2Nodes,
                selectedHidden2.map(item => item.value)
            );

            updateNodeValues(outputNodes, probabilities);

            // STEP 1

            status.textContent =
                `Using ${sample.source}. ` +
                "STEP 1: Input pixels enter the network.";

            activateNodes(inputNodes);

            await sleep(1000);

            await activateConnections("forward-1", "forward");

            activateNodes(hidden1Nodes);

            // STEP 2

            await sleep(1300);

            status.textContent =
                "STEP 2: Hidden Layer 1 applies weighted sums and ReLU.";

            await sleep(1000);

            activateConnections("forward-2", "forward");

            activateNodes(hidden2Nodes);

            // STEP 3

            await sleep(1300);

            status.textContent =
                "STEP 3: Hidden Layer 2 extracts further features using ReLU.";

            await sleep(1000);

            await activateConnections("forward-3", "forward");

            activateNodes(outputNodes);

            // STEP 4

            await sleep(1300);

            const predicted = probabilities.indexOf(
                Math.max(...probabilities)
            );

            status.textContent =
                `STEP 4: Softmax calculates output probabilities. ` +
                `Predicted digit: ${predicted} ` +
                `(${(probabilities[predicted] * 100).toFixed(2)}%).`;

        } catch (error) {

            console.error("Forward propagation:", error);

            status.textContent =
                "Visualization error: " + error.message;

        } finally {

            if (inputTensor) inputTensor.dispose();
            if (hidden1Tensor) hidden1Tensor.dispose();
            if (hidden2Tensor) hidden2Tensor.dispose();
            if (outputTensor) outputTensor.dispose();

            running = false;

            forwardBtn.disabled = false;
            backwardBtn.disabled = false;
            resetBtn.disabled = false;

        }

    }

    // ======================================
    // BACKPROPAGATION
    // ======================================

    
async function backwardPropagation() {

    if (running) return;

    if (!isModelTrained || !model || !mnistData) {

        status.textContent =
            "Please train the ANN first before demonstrating backpropagation.";

        return;
    }

    running = true;

    forwardBtn.disabled = true;
    backwardBtn.disabled = true;
    resetBtn.disabled = true;

    clearAnimation();

    let batch;
    let predictionBefore;
    let predictionAfter;

    try {

        // STEP 1: Get a training sample

        status.textContent =
            "Preparing a training sample from MNIST...";

        batch = mnistData.nextTrainBatch(1);

        const labels = Array.from(batch.labels.dataSync());

        const actualLabel = labels.indexOf(
            Math.max(...labels)
        );

        // STEP 2: Make prediction BEFORE updating weights

        predictionBefore = model.predict(batch.xs);

        const probabilitiesBefore =
            Array.from(predictionBefore.dataSync());

        const predictedBefore = probabilitiesBefore.indexOf(
            Math.max(...probabilitiesBefore)
        );

        // Calculate actual categorical cross-entropy loss

        let lossBefore = 0;

        for (let i = 0; i < labels.length; i++) {

            lossBefore -= labels[i] *
                Math.log(probabilitiesBefore[i] + 1e-7);

        }

        predictionBefore.dispose();
        predictionBefore = null;

        updateNodeValues(outputNodes, probabilitiesBefore);

        activateNodes(outputNodes);

        status.textContent =
            `STEP 1: Actual digit: ${actualLabel}. ` +
            `Predicted digit: ${predictedBefore}.`;

        await sleep(1500);

        // STEP 3: Explain prediction error

        status.textContent =
            `STEP 2: Calculated prediction loss: ${lossBefore.toFixed(4)}. ` +
            "The network needs to adjust its weights.";

        await sleep(1500);

        // STEP 4: Backpropagation - OUTPUT TO HIDDEN 2

        status.textContent =
            "STEP 3: Error gradients travel backwards from Output to Hidden Layer 2.";

        await activateConnections(
            "forward-3",
            "backward"
        );

        activateNodes(hidden2Nodes);

        await sleep(800);

        // STEP 5: HIDDEN 2 TO HIDDEN 1

        status.textContent =
            "STEP 4: Gradients continue backwards through Hidden Layer 2 to Hidden Layer 1.";

        await activateConnections(
            "forward-2",
            "backward"
        );

        activateNodes(hidden1Nodes);

        await sleep(800);

        // STEP 6: HIDDEN 1 TO INPUT

        status.textContent =
            "STEP 5: Backpropagation reaches the input side of the network.";

        await activateConnections(
            "forward-1",
            "backward"
        );

        activateNodes(inputNodes);

        await sleep(1200);

        // STEP 7: ACTUAL WEIGHT UPDATE

        status.textContent =
            "STEP 6: TensorFlow.js calculates gradients and updates the actual ANN weights...";

        const trainingResult = await model.trainOnBatch(
            batch.xs,
            batch.labels
        );

        // STEP 8: Check prediction after weight update

        predictionAfter = model.predict(batch.xs);

        const probabilitiesAfter =
            Array.from(predictionAfter.dataSync());

        let lossAfter = 0;

        for (let i = 0; i < labels.length; i++) {

            lossAfter -= labels[i] *
                Math.log(probabilitiesAfter[i] + 1e-7);

        }

        predictionAfter.dispose();
        predictionAfter = null;

        updateNodeValues(outputNodes, probabilitiesAfter);

        const newPrediction = probabilitiesAfter.indexOf(
            Math.max(...probabilitiesAfter)
        );

        const resultLoss = Array.isArray(trainingResult)
            ? Number(trainingResult[0])
            : Number(trainingResult);

        // FINAL RESULT

        status.textContent =
            `BACKPROPAGATION COMPLETED! ` +
            `Loss before update: ${lossBefore.toFixed(4)} | ` +
            `Loss after update: ${lossAfter.toFixed(4)} | ` +
            `New prediction: ${newPrediction}. ` +
            `Training loss: ${resultLoss.toFixed(4)}.`;

        console.log("Backpropagation completed.");
        console.log("Actual digit:", actualLabel);
        console.log("Prediction before:", predictedBefore);
        console.log("Prediction after:", newPrediction);
        console.log("Loss before:", lossBefore);
        console.log("Loss after:", lossAfter);

    } catch (error) {

        console.error("Backpropagation:", error);

        status.textContent =
            "Backpropagation error: " + error.message;

    } finally {

        if (predictionBefore) predictionBefore.dispose();
        if (predictionAfter) predictionAfter.dispose();

        if (batch) {
            batch.xs.dispose();
            batch.labels.dispose();
        }

        running = false;

        forwardBtn.disabled = false;
        backwardBtn.disabled = false;
        resetBtn.disabled = false;

    }

}


    // ======================================
    // RESET VISUALIZATION
    // ======================================

    function resetVisualization() {

        if (running) return;

        clearAnimation();

        inputNodes.forEach(node => {
            node.value.textContent = "0.00";
        });

        hidden1Nodes.forEach(node => {
            node.value.textContent = "0.00";
        });

        hidden2Nodes.forEach(node => {
            node.value.textContent = "0.00";
        });

        outputNodes.forEach((node, i) => {
            node.value.textContent = String(i);
        });

        status.textContent =
            "Visualization reset. Ready for another demonstration.";

    }

    // Button events

    forwardBtn.addEventListener(
        "click",
        forwardPropagation
    );

    backwardBtn.addEventListener(
        "click",
        backwardPropagation
    );

    resetBtn.addEventListener(
        "click",
        resetVisualization
    );

    // Build the network

    buildNetwork();

    console.log("Four-layer ANN visualization initialized.");

    console.log("Architecture: 784 → 128 → 64 → 10");

    console.log("Displayed connections:", edges.length);

})();



// ======================================
// INITIALIZE INTERACTIVE DIGIT DRAWING
// ======================================

const drawingGrid = document.getElementById("drawing-grid");
const clearDrawingBtn = document.getElementById("clear-drawing-btn");
const predictDigitBtn = document.getElementById("predict-digit-btn");
const previewCanvas = document.getElementById("processed-preview-canvas");

// Store 784 pixels (28 x 28)
let drawnPixels = new Array(784).fill(0);
let isDrawing = false;

// Create the 28 x 28 drawing grid
if (drawingGrid) {

    drawingGrid.innerHTML = "";

    for (let i = 0; i < 784; i++) {

        const pixel = document.createElement("div");

        pixel.className = "pixel";
        pixel.dataset.index = i;

        drawingGrid.appendChild(pixel);
    }
}

// Activate a drawing pixel

function activatePixel(pixel) {

    if (!pixel || !pixel.classList.contains("pixel")) return;

    const index = Number(pixel.dataset.index);

    const row = Math.floor(index / 28);
    const col = index % 28;

    // Create a 3 x 3 drawing brush
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


// Initialize processed preview
function clearProcessedPreview() {

    if (!previewCanvas) return;

    const ctx = previewCanvas.getContext("2d");

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
}

clearProcessedPreview();

console.log("Drawing grid initialized: 784 pixels.");


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
            isModelTrained
    ? "Draw a digit to get started."
    : "Please train the ANN first before predicting.";

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

    const cellSize = 20;

    previewCanvas.width = 28 * cellSize;
    previewCanvas.height = 28 * cellSize;

    const imageData = outputCtx.getImageData(0, 0, 28, 28);

    previewCtx.fillStyle = "black";
    previewCtx.fillRect(0, 0, 560, 560);

    previewCtx.textAlign = "center";
    previewCtx.textBaseline = "middle";
    previewCtx.font = "bold 10px Arial";

    for (let y = 0; y < 28; y++) {

        for (let x = 0; x < 28; x++) {

            const index = (y * 28 + x) * 4;

            const value = imageData.data[index];

            const pixelValue = Math.round(value);

            const posX = x * cellSize;
            const posY = y * cellSize;

            // Draw grayscale pixel
            previewCtx.fillStyle =
                `rgb(${value}, ${value}, ${value})`;

            previewCtx.fillRect(
                posX,
                posY,
                cellSize,
                cellSize
            );

            // Choose readable text colour
            previewCtx.fillStyle =
                value > 130 ? "black" : "white";

            // Display actual pixel intensity
            previewCtx.fillText(
                pixelValue,
                posX + cellSize / 2,
                posY + cellSize / 2
            );

            // Pixel borders
            previewCtx.strokeStyle = "#555";
            previewCtx.lineWidth = 0.5;

            previewCtx.strokeRect(
                posX,
                posY,
                cellSize,
                cellSize
            );
        }
    }
}


    return pixels;
}


// =====================================
// ACTUAL ANN DIGIT PREDICTION
// =====================================


if (predictDigitBtn) {

    predictDigitBtn.addEventListener("click", () => {

        if (!isModelTrained) {
            document.getElementById("drawing-status").textContent =
                "Please train the ANN first!";
            return;
        }

        if (!drawnPixels.some(pixel => pixel > 0)) {
            document.getElementById("drawing-status").textContent =
                "Please draw a digit first!";
            return;
        }

        try {

            // Preprocess YOUR drawing
            const processedPixels = preprocessDrawnPixels();

            // Pass drawing through the trained ANN
            const probabilities = tf.tidy(() => {

                const input = tf.tensor2d(
                    processedPixels,
                    [1, 784]
                );

                const prediction = model.predict(input);

                return Array.from(prediction.dataSync());

            });

            // Find highest probability
            let predictedDigit = 0;

            for (let i = 1; i < 10; i++) {
                if (probabilities[i] > probabilities[predictedDigit]) {
                    predictedDigit = i;
                }
            }

            const confidence = probabilities[predictedDigit] * 100;

            // Display predicted digit
            document.getElementById("predicted-digit").textContent =
                predictedDigit;

            document.getElementById("drawing-status").textContent =
                "Prediction completed successfully!";

            document.getElementById("prediction-confidence").textContent =
                "Confidence: " + confidence.toFixed(2) + "%";

            // Create probability dashboard
            const resultContainer =
                document.getElementById("drawing-result");

            let probabilitySection =
                document.getElementById("drawing-probabilities");

            if (!probabilitySection) {

                probabilitySection = document.createElement("div");

                probabilitySection.id = "drawing-probabilities";

                resultContainer.appendChild(probabilitySection);
            }

            probabilitySection.innerHTML = `
                <h3>Prediction Probabilities (0–9)</h3>
                <div class="probability-grid"></div>
            `;

            const grid =
                probabilitySection.querySelector(".probability-grid");

            probabilities.forEach((probability, digit) => {

                const percentage = probability * 100;

                const item = document.createElement("div");

                item.className = "probability-item";

                item.innerHTML = `
                    <strong>${digit}</strong>
                    <div class="probability-bar">
                        <div class="probability-fill"
                             style="width: ${percentage}%">
                        </div>
                    </div>
                    <span>${percentage.toFixed(2)}%</span>
                `;

                if (digit === predictedDigit) {
                    item.classList.add("predicted");
                }

                grid.appendChild(item);

            });

            console.log("Your drawing prediction:", predictedDigit);
            console.log("Probabilities:", probabilities);

        } catch (error) {

            console.error("Prediction error:", error);

            document.getElementById("drawing-status").textContent =
                "Prediction error: " + error.message;

        }

    });

}

// ======================================
// IMAGE TO NUMERICAL INPUT VISUALIZATION
// ======================================

const viewPixelValuesBtn =
    document.getElementById("view-pixel-values-btn");

const pixelValuesPanel =
    document.getElementById("pixel-values-panel");

const pixelValuesGrid =
    document.getElementById("pixel-values-grid");

if (viewPixelValuesBtn) {

    viewPixelValuesBtn.addEventListener("click", () => {

        // Toggle panel visibility
        if (pixelValuesPanel.style.display === "block") {

            pixelValuesPanel.style.display = "none";

            viewPixelValuesBtn.textContent =
                "View Pixel Values";

            return;
        }

        // Check whether user has drawn something
        if (!drawnPixels.some(pixel => pixel > 0)) {

            document.getElementById("drawing-status")
                .textContent = "Please draw a digit first!";

            return;
        }

        // Get actual processed ANN input
        const pixels = preprocessDrawnPixels();

        // Clear previous values
        pixelValuesGrid.innerHTML = "";

        // Display all 784 numerical values
        pixels.forEach((value, index) => {

            const pixel = document.createElement("div");

            pixel.className = "pixel-value";

            pixel.textContent = value.toFixed(2);

            pixel.title =
                `Input ${index + 1} | Row ${Math.floor(index / 28) + 1} | Column ${(index % 28) + 1}`;

            pixelValuesGrid.appendChild(pixel);

        });

        pixelValuesPanel.style.display = "block";

        viewPixelValuesBtn.textContent =
            "Hide Pixel Values";

    });

}
