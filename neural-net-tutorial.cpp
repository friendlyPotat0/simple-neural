// neural-net-tutorial.cpp

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

using std::ifstream;
using std::mt19937;
using std::random_device;
using std::string;
using std::stringstream;
using std::uniform_int_distribution;
using std::vector;

// Doofus class to read training data from file

class TrainingData {
  public:
    TrainingData(const string filename);
    bool isEOF() { return training_data_file.eof(); }
    void get_topology(vector<unsigned> &topology);
    // Returns number of input values read from file
    unsigned get_next_inputs(vector<double> &input_values);
    unsigned get_target_outputs(vector<double> &target_values);

  private:
    ifstream training_data_file;
};

TrainingData::TrainingData(const string filename) { training_data_file.open(filename.c_str()); }

void TrainingData::get_topology(vector<unsigned> &topology) {
    string line;
    string label;
    getline(training_data_file, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEOF() || label.compare("topology:") != 0) {
        abort();
    }
    while (!ss.eof()) {
        unsigned i;
        ss >> i;
        topology.push_back(i);
    }
    return; // !!
}

unsigned TrainingData::get_next_inputs(vector<double> &input_values) {
    input_values.clear();

    string line;
    getline(training_data_file, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double one_value;
        while (ss >> one_value) {
            input_values.push_back(one_value);
        }
    }
    return input_values.size();
}

unsigned TrainingData::get_target_outputs(vector<double> &target_values) {
    target_values.clear();

    string line;
    getline(training_data_file, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double one_value;
        while (ss >> one_value) {
            target_values.push_back(one_value);
        }
    }

    return target_values.size();
}

struct Connection {
    double weight; // output weights in the neuron to the right
    double delta_weight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
  public:
    Neuron(unsigned outputs_number, unsigned index);
    void set_output_value(const double input_value) { output_value = input_value; } // !!
    double get_output_value() const { return output_value; }
    void feed_forward(const Layer &previous_layer);
    void calculate_output_gradients(double target_value);
    void calculate_hidden_gradients(const Layer &next_layer);
    void update_input_weights(Layer &previous_layer);

  private:
    /* [0.0, 1.0]
    overall neural network training/learning rate */
    static double eta;
    /* [0.0, n]
    alpha (momentum) = a fraction of the previous delta weight
    allows the optimizer to continue advancing in the same direction as
    previously, minimizing oscillations and increasing convergence speed */
    static double alpha;
    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    static double random_weight();
    double sumDOW(const Layer &next_layer) const;
    double output_value;
    vector<Connection> connections;
    unsigned index;
    double gradient;
};

Neuron::Neuron(unsigned outputs_number, unsigned index) {
    for (unsigned i = 0; i < outputs_number; ++i) {
        connections.push_back(Connection());
        connections.back().weight = random_weight();
    }
    this->index = index;
}

// tweak these values!
/* 0.0 -> slow learner
0.2 -> medium learner
1.0 -> reckless learner */
double Neuron::eta = 0.15;
/* 0.0 -> no momentum
0.5 -> moderate momentum */
double Neuron::alpha = 0.5;

double Neuron::random_weight() {
    mt19937 rng(random_device{}());
    uniform_int_distribution<> dist(0, 1);
    return dist(rng);
}

double Neuron::transfer_function(double x) {
    // output range [-1.0, 1.0]
    return tanh(x);
}

double Neuron::transfer_function_derivative(double x) {
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::update_input_weights(Layer &previous_layer) {
    // The weights to be updated are in the connnection container of the neurons
    // of the previous layer
    for (unsigned i = 0, n = previous_layer.size(); i < n; ++i) {
        double previous_delta_weight = previous_layer.at(i).connections.at(index).delta_weight;
        double current_delta_weight = eta * previous_layer.at(i).get_output_value() * gradient + alpha * previous_delta_weight;
        previous_layer.at(i).connections.at(index).delta_weight = current_delta_weight;
        previous_layer.at(i).connections.at(index).weight += current_delta_weight;
    }
}

double Neuron::sumDOW(const Layer &next_layer) const {
    double sum = 0.0;
    // Sum contributions of the errors at the nodes that are fed
    for (unsigned i = 0, n = next_layer.size() - 1; i < n; ++i) {
        sum += connections.at(i).weight * next_layer.at(i).gradient;
    }
    return sum;
}

void Neuron::feed_forward(const Layer &previous_layer) {
    double sum = 0.0;
    // Sum the previous layer's outputs
    // Include the bias node from the previous layer
    for (unsigned i = 0, n = previous_layer.size(); i < n; ++i) {
        sum += previous_layer.at(i).get_output_value() * previous_layer.at(i).connections.at(index).weight;
    }
    output_value = Neuron::transfer_function(sum);
}

void Neuron::calculate_output_gradients(double target_value) {
    double delta = target_value - output_value;
    gradient = delta * Neuron::transfer_function_derivative(output_value);
}

void Neuron::calculate_hidden_gradients(const Layer &next_layer) {
    double dow = sumDOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output_value);
}

class NeuralNetwork {
  public:
    NeuralNetwork(const vector<unsigned> &topology);
    void feed_forward(const vector<double> &input_values); // AKA train
    void back_propagation(const vector<double> &target_values);
    void get_results(vector<double> &result_values) const;
    double get_recent_average_error() const { return recent_average_error; }

  private:
    vector<Layer> layers;
    double error;
    double recent_average_error;
    double recent_average_smoothing_factor;
};

NeuralNetwork::NeuralNetwork(const vector<unsigned> &topology) {
    for (unsigned i = 0, n = topology.size(); i < n; ++i) {
        layers.push_back(Layer());
        unsigned outputs_number = i == topology.size() - 1 ? 0 : topology.at(i + 1);
        for (unsigned j = 0, n = topology.at(i); j <= n; ++j) { // <= -> includes bias layer
            layers.back().push_back(Neuron(outputs_number, j));
            printf("Neuron created!\n");
        }
        // Force the bias node's output value to 1.0
        layers.back().back().set_output_value(1.0);
    }
}

void NeuralNetwork::get_results(vector<double> &result_values) const {
    result_values.clear();
    for (unsigned i = 0, n = layers.back().size() - 1; i < n; ++i) {
        result_values.push_back(layers.back().at(i).get_output_value());
    }
}

void NeuralNetwork::feed_forward(const vector<double> &input_values) {
    assert(input_values.size() == layers.at(0).size() - 1);
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0, n = input_values.size(); i < n; ++i) {
        layers.at(0).at(i).set_output_value(input_values.at(i));
    }
    // Forward propagation
    for (unsigned i = 1, n = layers.size(); i < n; ++i) {
        Layer &previous_layer = layers.at(i - 1);
        for (unsigned j = 0; j < layers.at(i).size() - 1; ++j) {
            layers.at(i).at(j).feed_forward(previous_layer);
        }
    }
}
void NeuralNetwork::back_propagation(const vector<double> &target_values) {
    // Calculate overall neural network error (RMSE [root mean square error] of
    // output neuron errors)
    Layer &output_layer = layers.back();
    error = 0.0;
    for (unsigned i = 0, n = output_layer.size() - 1; i < n; ++i) {
        double delta = target_values.at(i) - output_layer.at(i).get_output_value();
        error += delta * delta;
    }
    error /= output_layer.size() - 1;
    error = sqrt(error);
    recent_average_error = (recent_average_error * recent_average_smoothing_factor + error) / (recent_average_smoothing_factor + 1.0);
    // Calculate output layer gradients
    for (unsigned i = 0, n = output_layer.size() - 1; i < n; ++i) {
        output_layer.at(i).calculate_output_gradients(target_values.at(i));
    }
    // Calculate gradients on hidden layers
    for (unsigned i = layers.size() - 2; i > 0; --i) {
        Layer &current_layer = layers.at(i);
        Layer &next_layer = layers.at(i + 1);
        for (unsigned j = 0, n = current_layer.size(); j < n; ++j) {
            current_layer.at(j).calculate_hidden_gradients(next_layer);
        }
    }
    // For all layers from outputs to first hidden layer, update connection
    // weights
    for (unsigned i = layers.size() - 1; i > 0; --i) {
        Layer &current_layer = layers.at(i);
        Layer &previous_layer = layers.at(i - 1);
        for (unsigned j = 0, n = current_layer.size() - 1; j < n; ++j) {
            current_layer.at(j).update_input_weights(previous_layer);
        }
    }
}

void show_vector_values(string label, vector<double> &vec) {
    printf("%s ", label.c_str());
    for (unsigned i = 0, n = vec.size(); i < n; ++i) {
        printf("%.1f ", vec.at(i));
    }
    printf("\n");
}

int main() {
    TrainingData training_data("training-data.txt");

    vector<unsigned> topology;
    training_data.get_topology(topology);
    NeuralNetwork neural_network(topology);

    vector<double> input_values, target_values, result_values;
    int training_pass = 0;
    while (!training_data.isEOF()) {
        ++training_pass;
        printf("PASS %i\n", training_pass);
        // Get new input data and feed it forward
        if (training_data.get_next_inputs(input_values) != topology.at(0)) {
            break;
        }
        show_vector_values("Inputs:", input_values);
        neural_network.feed_forward(input_values);
        // Collect the neural network results:
        neural_network.get_results(result_values);
        show_vector_values("Outputs:", result_values);
        // Train the network based on what the outputs should have been
        training_data.get_target_outputs(target_values);
        show_vector_values("Targets:", target_values);
        assert(target_values.size() == topology.back());
        neural_network.back_propagation(target_values);
        // Report learning progress
        printf("Recent average error (RMSE): %f\n\n", neural_network.get_recent_average_error());
    }
    printf("\nDone!\n");
    return 0;
}
