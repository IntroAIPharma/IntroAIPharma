# Chemistry
import rdkit
from rdkit import Chem

# Property is the abstract class from which all
# properties must inherit.
from Property import Property

# Local
from CHEMBERT.chembert import chembert_model, SMILES_Dataset

class CHEMBERT_predictor(Property):
    """
        Calculator class for CHEM-BERT 
        Estimates a property given a CHEM-BERT model and a SMILES file.

        Note: This calculates the property of only ONE molecule, which is definately *not*
              the most efficient use of the CHEM-BERT implementation, but is one that
              fits our model. Later, we may come back to a more efficient implementation.
    """

    CITATION = ( "  Kim, H., Lee, J., Ahn, S., & Lee, J. R. (2021).\n"
                 "  \"A merged molecular representation learning for molecular \n"
                 "  properties prediction with a web-based service.\" \n"
                 "  Scientific Reports, 11(1), 11028.\n"
                 "  https://doi.org/10.1038/s41598-021-90259-7" )

    def __init__(self, prop_name, **kwargs):
        # Initialize super
        print("Initializing CHEMBERT model")
        print(f"   Property Name: {prop_name}")
        for key,value in kwargs.items():
            print(f"   {key}: {value}")
        super().__init__(prop_name, **kwargs)
        self.model_file = kwargs['model_file']
        self.model = chembert_model(self.model_file, task=kwargs['task'])
        print("Done.")

    def predict(self, 
                smis,
                **kwargs):
        """
            Args:
                smis (SMILES or list): molecule(s) to be evaluated

            Returns:
                float: Predicted LogBB value
        """

        _smis, chembert_scores = [], []
        _smis.extend(smis)

        if not isinstance(_smis, str):
            _smis = [Chem.MolToSmiles(x) for x in _smis]
        dataset = SMILES_Dataset(_smis)
        chembert_scores = self.model.predict(dataset)

        return chembert_scores
