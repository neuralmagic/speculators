import torch 

# understand the different variations from
# https://github.com/SafeAILab/EAGLE

class DraftModel:
    """"""
class VerifierModel:
    pass

class SpeculatorModel:
    """
    Ties together the speculator, proposal algo, etc
    """

    draft_model: DraftModel
    verifier_model: VerifierModel

    def save_pretrained(self, *args, **kwargs):
        """
        Save the speculator model
        """
        # save the config, ()
        # DO NOT save the verifier model
        # save the drafter model

        # Save name of the proposal algo
            # Extra arguments for proposal algo like top_k?
        pass



speculator_config = {
    # drafter
    # sampling algorithm (proposal)
    # verfication algorithm
    # point to the verifier model
}


speculator_model = SpeculatorModel.from_config(speculator_config)
speculator_model.save_pretrained() 
# modify save_pretrained to save the speculator model
# model.safetensor for the speculator, , config.json containing the speculator_config
# for eagle speculator need to be converted

"""
Lifecycle of a saving:

1) state_dict -> potentially a conversion step -> save to disk
Note: Some layers can be shared b/w speculator and verifier, try to
not save those layers (maybe have an argument for that on config)

The connection points(shared layers) should be in the SpeculatorConfig;

2) Consolidating the configs into a config.json

3) The saved checkpoint can be loaded in vLLM
"""