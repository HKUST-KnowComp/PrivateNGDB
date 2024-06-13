class PrivateModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def private_protect(self, ):
        pass