import numpy as np
from .neuron import Neuron

class Brain:
    def __init__(self, num_neurons=100, connection_prob=0.1):
        self.num_neurons = num_neurons
        self.neurons = [
            Neuron(i, neuron_type='inhibitory' if np.random.rand() < 0.2 else 'excitatory')
            for i in range(num_neurons)
        ]

        # Synaptic weights: normally distributed, clipped, and sparse
        self.weights = np.random.normal(0.5, 0.3, (num_neurons, num_neurons))
        mask = (np.random.rand(num_neurons, num_neurons) < connection_prob)
        self.weights *= mask
        self.weights = np.clip(self.weights, 0.0, 1.0)

        # Inhibitory neurons have negative outgoing weights
        for i, neuron in enumerate(self.neurons):
            if neuron.type == 'inhibitory':
                self.weights[i, :] *= -1

        # Synaptic delays (1-5 time steps)
        self.delays = np.random.randint(1, 6, size=(num_neurons, num_neurons)) * mask

        # Spike buffers for delays: list of lists per neuron
        self.spike_buffers = [[[] for _ in range(num_neurons)] for _ in range(max(self.delays.flatten())+1)]

        # Parameters for STDP
        self.stdp_tau_pre = 20  # ms (or steps)
        self.stdp_tau_post = 20
        self.stdp_A_plus = 0.01
        self.stdp_A_minus = -0.012

        # Track last spike times for STDP (initialize with large negative number)
        self.last_spike_times = -1000 * np.ones(num_neurons)

    def step(self, t):
        inputs = np.zeros(self.num_neurons)

        # Collect incoming spikes considering delays
        delay_max = len(self.spike_buffers)
        delayed_spikes = self.spike_buffers.pop(0)
        self.spike_buffers.append([[] for _ in range(self.num_neurons)])

        # Process delayed spikes into input currents
        for pre in range(self.num_neurons):
            for post in range(self.num_neurons):
                if delayed_spikes[post].count(pre) > 0:
                    inputs[post] += self.weights[pre, post]

        # Add small random background noise/input
        inputs += np.random.normal(0.05, 0.02, self.num_neurons)

        # Stimulate neurons and collect firings
        firings = []
        for i, neuron in enumerate(self.neurons):
            neuron.stimulate(inputs[i])
            if neuron.fired:
                firings.append(i)
                self.last_spike_times[i] = t

        # Enqueue spikes into buffers for postsynaptic neurons with delays
        for pre in firings:
            for post in range(self.num_neurons):
                delay = self.delays[pre, post]
                if delay > 0:
                    self.spike_buffers[delay].append([post])
                else:
                    # No connection or zero delay => immediate effect
                    inputs[post] += self.weights[pre, post]

        # STDP weight updates
        self.apply_stdp(t, firings)

        # Synaptic normalization: keep weights bounded and sum constraints
        self.weights = np.clip(self.weights, -1.0, 1.0)
        for i in range(self.num_neurons):
            pos_weights = self.weights[i, :] > 0
            total_pos = np.sum(self.weights[i, pos_weights])
            if total_pos > 1.5:
                self.weights[i, pos_weights] /= total_pos / 1.5

            neg_weights = self.weights[i, :] < 0
            total_neg = np.sum(np.abs(self.weights[i, neg_weights]))
            if total_neg > 1.0:
                self.weights[i, neg_weights] /= total_neg / 1.0

        return [(t, i) for i in firings]

    def apply_stdp(self, t, firings):
        # Simple STDP rule based on spike timing difference
        for post in firings:
            for pre in range(self.num_neurons):
                if pre == post:
                    continue
                dt = self.last_spike_times[post] - self.last_spike_times[pre]
                if dt > 0 and dt < self.stdp_tau_pre:
                    self.weights[pre, post] += self.stdp_A_plus * np.exp(-dt / self.stdp_tau_pre)
                elif dt < 0 and abs(dt) < self.stdp_tau_post:
                    self.weights[pre, post] += self.stdp_A_minus * np.exp(dt / self.stdp_tau_post)
