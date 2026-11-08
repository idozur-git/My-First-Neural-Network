def one_full_layer(inputs, weights, biases):
    '''Receives a list of input values of integers, a list of lists of weights for each neuron, and a bias value for
    each of the neurons. Outputs the fully connected result of the layer'''
    layer_outputs = [] #will be the result
    for neuron_weights, neuron_bias in zip(weights, biases): #iterating over lists of weights and biases
        neuron_output = 0 #initalize neuron n's output
        for input, weight in zip(inputs, neuron_weights): #iterating over inputs and each neuron's weights
            neuron_output += input*weight #each input multiplied by the weight with respect to the corresponding neuron
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    return layer_outputs
