class DecayingValue:
    def __init__(self, initial, target, changeFn):
        self.initial = initial
        self.value = self.initial
        self.target = target
        self.changeFn = changeFn
        self.values = []
        self.delegate = self

    def reset(self):
        self.value = self.initial

    def next(self):
        if self.value <= self.target:
            self.values.append(self.target)
            return self.target

        self.value = self.changeFn(self.value)
        if self.value < self.target:
            self.value = self.target

        self.values.append(self.value)
        return self.value

    def __add__(self, other):
        return self.next() + other

    def __mul__(self, other):
        return self.next() * other


class PositiveMemoriesFactorExplorationDecay(DecayingValue):
    def __init__(self, initial, target, gamma, lower_treshold, memory):
        super(PositiveMemoriesFactorExplorationDecay, self).__init__(initial, target, self.next_value)
        self.lower_treshold = lower_treshold
        self.gamma = gamma
        self.memory = memory

    def next_value(self, x):
        if self.memory.neg_count == 0:
            return x

        factor = self.memory.pos_count / self.memory.neg_count
        value = x
        if factor < self.lower_treshold:
            value = x + self.gamma
        else:
            value = x - self.gamma

        return max(0.0, min(1.0, value))

    def next(self):
        self.value = self.changeFn(self.value)
        if self.value < self.target:
            self.value = self.target

        self.values.append(self.value)
        return self.value


class ExponentialDecay(DecayingValue):
    def __init__(self, initial, target, gamma):
        super(ExponentialDecay, self).__init__(initial, target, lambda x: x * gamma)


class SpacedRepetitionDecay:
    def __init__(self, delegate, reset_initial, reset_gamma):
        self.delegate = delegate

        self.step = 0
        self.reset_initial = reset_initial
        self.reset_gamma = reset_gamma
        self.reset_step = self.reset_initial

    def reset(self):
        self.delegate.reset()
        self.step = 0
        self.reset_step = self.reset_initial

    def next(self):
        self.step += 1
        if self.step >= self.reset_step:
            self.delegate.reset()
            self.step = 0
            self.reset_step = self.reset_step * self.reset_gamma

        return self.delegate.next()
