#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;

double randomWeight(){
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<>dist(0,sqrt(2.0));
    return dist(gen);
}

void initializeWeights(vector<vector<double>>& hiddenWeights1, vector<vector<double>>& hiddenWeights2, vector<vector<double>>& outputWeights){
    for (size_t i=0; i<hiddenWeights1.size(); i++){
        for (size_t j=0; j<hiddenWeights1[i].size();j++){
            hiddenWeights1[i][j] = randomWeight();
        }
    }

    for (size_t i=0; i<hiddenWeights2.size(); i++){
        for (size_t j=0; j<hiddenWeights2[i].size();j++){
            hiddenWeights2[i][j] = randomWeight();
        }
    }

    for (size_t i=0; i<outputWeights.size(); i++){
        for (size_t j=0; j<outputWeights[i].size(); j++){
            outputWeights[i][j] = randomWeight();
        }
    }
}

// ReLU aktivasyon fonksiyonu
double relu(double x) {
    return (x > 0) ? x : 0;
}
double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Sigmoid aktivasyon fonksiyonu ve türevi
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

// Tek bir nöronun hesaplanması
double neuronOutput(const vector<double>& inputs, const vector<double>& weights, double bias) {
    double sum = bias;
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    return relu(sum);
}

// Çoklu nöron hesaplayan fonksiyon
vector<double> layerOutput(const vector<double>& inputs, const vector<vector<double>>& weights, const vector<double>& biases) {
    vector<double> outputs;
    for (size_t i = 0; i < weights.size(); i++) {
        outputs.push_back(neuronOutput(inputs, weights[i], biases[i]));
    }
    return outputs;
}

void clearScreen() {
    cout << "\033[2J\033[H";
}

void visualizeNetwork(const vector<double>& inputs, const vector<double>& hiddenLayer1, const vector<double>& hiddenLayer2, double output) {
    clearScreen();
    cout << "\n";
    cout << "   Input Layer\n";
    cout << "  -------------\n";
    for (double val : inputs) {
        cout << "  (O) " << val << "\n";
    }
    cout << "    |\n    v\n";
    
    cout << "   Hidden Layer 1\n";
    cout << "  --------------\n";
    for (double val : hiddenLayer1) {
        cout << "  " << (val > 0 ? "(O) " + to_string(val).substr(0, 4) : "(.)") << "\n";
    }
    cout << "    |\n    v\n";
    
    cout << "   Hidden Layer 2\n";
    cout << "  --------------\n";
    for (double val : hiddenLayer2) {
        cout << "  " << (val > 0 ? "(O) " + to_string(val).substr(0, 4) : "(.)") << "\n";
    }
    cout << "    |\n    v\n";
    
    cout << "   Output Layer\n";
    cout << "  --------------\n";
    cout << "  (O) " << output << "\n";
    
    this_thread::sleep_for(chrono::milliseconds(300));
}

vector<double> feedForward(
    const vector<double>& inputs,
    const vector<vector<double>>& hiddenWeights1,
    const vector<double>& hiddenBiases1, 
    const vector<vector<double>>& hiddenWeights2,
    const vector<double>& hiddenBiases2, 
    const vector<vector<double>>& outputWeights, 
    const vector<double>& outputBiases
){
    vector<double> hiddenLayer1 = layerOutput(inputs, hiddenWeights1, hiddenBiases1);
    vector<double> hiddenLayer2 = layerOutput(hiddenLayer1, hiddenWeights2, hiddenBiases2);
    vector<double> finalOutput;
    for(size_t i=0; i<outputWeights.size(); i++){
        double sum = outputBiases[i];
        for (size_t j=0; j<hiddenLayer2.size(); j++){
            sum += hiddenLayer2[j] * outputWeights[i][j];
        }
        finalOutput.push_back(sigmoid(sum));
    }
    visualizeNetwork(inputs, hiddenLayer1, hiddenLayer2, finalOutput[0]);
    return finalOutput;
}

void backpropagation(
    const vector<double>& inputs,
    const double targetOutput,
    vector<vector<double>>& hiddenWeights1,
    vector<double>& hiddenBiases1,
    vector<vector<double>>& hiddenWeights2,
    vector<double>& hiddenBiases2,
    vector<vector<double>>& outputWeights,
    vector<double>& outputBiases,
    double learningRate
) {
    vector<double> hiddenLayer1 = layerOutput(inputs, hiddenWeights1, hiddenBiases1);
    vector<double> hiddenLayer2 = layerOutput(hiddenLayer1, hiddenWeights2, hiddenBiases2);
    vector<double> finalOutput = feedForward(inputs, hiddenWeights1, hiddenBiases1, hiddenWeights2, hiddenBiases2, outputWeights, outputBiases);
    
    double outputError = targetOutput - finalOutput[0];
    double outputDelta = outputError * sigmoidDerivative(finalOutput[0]);
    
    vector<double> hiddenDeltas2(hiddenLayer2.size());
    for (size_t i = 0; i < hiddenLayer2.size(); i++) {
        hiddenDeltas2[i] = outputDelta * outputWeights[0][i] * reluDerivative(hiddenLayer2[i]);
    }
    
    vector<double> hiddenDeltas1(hiddenLayer1.size());
    for (size_t i = 0; i < hiddenLayer1.size(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < hiddenLayer2.size(); j++) {
            sum += hiddenDeltas2[j] * hiddenWeights2[j][i];
        }
        hiddenDeltas1[i] = sum * reluDerivative(hiddenLayer1[i]);
    }
    
    for (size_t i = 0; i < outputWeights.size(); i++) {
        for (size_t j = 0; j < hiddenLayer2.size(); j++) {
            outputWeights[i][j] += learningRate * outputDelta * hiddenLayer2[j];
        }
        outputBiases[i] += learningRate * outputDelta;
    }
    
    for (size_t i = 0; i < hiddenWeights2.size(); i++) {
        for (size_t j = 0; j < hiddenLayer1.size(); j++) {
            hiddenWeights2[i][j] += learningRate * hiddenDeltas2[i] * hiddenLayer1[j];
        }
        hiddenBiases2[i] += learningRate * hiddenDeltas2[i];
    }
    
    for (size_t i = 0; i < hiddenWeights1.size(); i++) {
        for (size_t j = 0; j < inputs.size(); j++) {
            hiddenWeights1[i][j] += learningRate * hiddenDeltas1[i] * inputs[j];
        }
        hiddenBiases1[i] += learningRate * hiddenDeltas1[i];
    }
}

int main() {
    // Veri kümesi oluştur
    vector<vector<double>> inputs = {
        {1.0, 2.0, 3.0, 4.0},
        {0.5, 1.5, 2.5, 3.5},
        {1.5, 2.5, 3.5, 4.5},
        {0.8, 1.8, 2.8, 3.8}
    };
    vector<double> targets = {0.05, 0.10, 0.15, 0.20}; // Hedef çıktılar

    // Ağırlık ve biasları tanımla
    vector<vector<double>> hiddenWeights1 = {
        {0.5, -1.5, 2.0, 0.8}, 
        {1.0, 2.0, -1.0, -0.5}
    };
    vector<double> hiddenBiases1 = {0.5, -1.0};
    
    vector<vector<double>> hiddenWeights2 = {
        {0.5, -1.5}, 
        {1.0, 2.0}
    };
    vector<double> hiddenBiases2 = {0.5, -1.0};
    
    vector<vector<double>> outputWeights = {{0.7, -1.2}};
    vector<double> outputBiases = {0.3};
    
    double learningRate = 0.01;
    int batchSize = 2; // Mini-batch boyutu
    int epochs = 1000;

    initializeWeights(hiddenWeights1, hiddenWeights2, outputWeights);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Veri kümesini karıştır
        vector<size_t> indices(inputs.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
        shuffle(indices.begin(), indices.end(), mt19937(random_device()()));

        // Mini-batch'ler üzerinde iterasyon
        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            // Mini-batch'i al
            size_t end = min(i + batchSize, inputs.size());
            for (size_t j = i; j < end; j++) {
                size_t idx = indices[j];
                backpropagation(inputs[idx], targets[idx], hiddenWeights1, hiddenBiases1, hiddenWeights2, hiddenBiases2, outputWeights, outputBiases, learningRate);
            }
        }
    }
    
    return 0;
}