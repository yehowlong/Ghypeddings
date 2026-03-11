class EarlyStoppingCriterion(object):
    """
    Arguments:
        patience (int): The maximum number of epochs with no improvement before early stopping should take place
        mode (str, can only be 'max' or 'min'): To take the maximum or minimum of the score for optimization
        min_delta (float, optional): Minimum change in the score to qualify as an improvement (default: 0.0)
    """

    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {'min', 'max'}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self.best_dev_score = None
        self.best_epoch = None
        self.is_improved = None
        self.best_emb = None

    def step(self, cur_dev_score, epoch , embeddings):
        """
        Checks if training should be continued given the current score.

        Arguments:
            cur_dev_score (float): the current development score
            cur_test_score (float): the current test score
        Output:
            bool: if training should be continued
        """
        if self.best_dev_score is None:
            self.best_dev_score = cur_dev_score
            self.best_epoch = epoch
            self.best_emb = embeddings
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_dev_score > self.best_dev_score + self.min_delta)
            else:
                self.is_improved = (cur_dev_score < self.best_dev_score - self.min_delta)

            if self.is_improved:
                self._count = 0
                self.best_dev_score = cur_dev_score
                self.best_epoch = epoch
                self.best_emb = embeddings
            else:
                self._count += 1
            return self._count <= self.patience
