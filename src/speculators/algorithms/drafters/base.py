

class DraftModel:
    """
    Base class for all draft models.

    Will be used for potentially modifying the drafter model
    """

    head = None
    body = None

    def modify_drafter(self, *args, **kwargs):
        """
        Modify the drafter model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def input_layers(self):
        """
        Will define which layer does the drafter model takes
        inputs from

        # different drafters can have different inputs, for example:
        # TODO: define the lifecycle of outputs based on the algorithm used
        # every drafter will have a method for clearing the cache after acceptance
        
        :param: List of layer names
        :param: Tensor shapes, # shapes of outputs from each layer
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def output_(self):
        """
        Defines which proposal
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    


class VerifierModel:
    """
    Base class for all verifier models.

    Can hold a reference to the verifier model
    (But DO NOT save the weight, try not to hold a reference to the model)
    """
    model_name: str
    configs: ...

    