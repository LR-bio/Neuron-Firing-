import numpy as np

class Neuron:
    def __init__(self, neuron_id, neuron_type='excitatory', base_threshold=1.0, decay=0.05, refractory_period=5):
        self.id = neuron_id
        self.type = neuron_type  # 'excitatory' or 'inhibitory'
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.decay = decay
        self.potential = 0.0
        self.fired = False
        self.refractory_period = refractory_period
        self.refractory_timer = 0

    def reset(self):
        self.fired = False

    def stimulate(self, input_current):
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.fired = False
            # Potential slightly decays even during refractory
            self.potential *= (1 - self.decay)
            return

        # Leaky integrate and fire with membrane noise
        noise = np.random.normal(0, 0.02)
        self.potential = self.potential * (1 - self.decay) + input_current + noise

        # Fire if over threshold
        if self.potential >= self.threshold:
            self.fired = True
            self.potential = 0.0
            self.refractory_timer = self.refractory_period
            self.threshold *= 1.10  # threshold increases after firing (adaptation)
        else:
            self.fired = False
            # Threshold homeostasis to baseline
            self.threshold += (self.base_threshold - self.threshold) * 0.01
