# set parameters
class Params():
    
    def __init__(
        self,
        device,
        n_output_labels = 2,
        lr = 2e-5,
        test_size = 0.2,
        n_splits = 5,
        n_epochs = 10,
        batch_size = 32,
        max_len = 100,
        betas = (0.9, 0.98),
        eps = 1e-8,
        random_state = 42
    ) -> None:
        self.device = device
        self.n_output_labels = n_output_labels
        self.lr = lr
        self.test_size = test_size
        self.n_splits = n_splits
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.betas = betas
        self.eps = eps
        self.random_state = random_state