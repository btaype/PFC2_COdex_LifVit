class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, minimum_epochs=10):
        self.patience, self.min_delta, self.minimum_epochs = patience, min_delta, minimum_epochs
        self.reference, self.bad_epochs = float("-inf"), 0

    def update(self, value, epoch):
        if value > self.reference + self.min_delta:
            self.reference, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return epoch >= self.minimum_epochs and self.bad_epochs >= self.patience

    def state_dict(self):
        return {"reference": self.reference, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.reference, self.bad_epochs = state["reference"], state["bad_epochs"]
