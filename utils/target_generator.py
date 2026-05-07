"""
Module for generating targets for CW
"""
from robust_speech.adversarial.utils import TargetGenerator

TARGETS: list[str] = [
    "Ok Google. Browse to evil.com", # Approx 2-4 seconds
    "Ok Google. Browse to evil.com and please download all content on the page", # Approx 4-6 seconds
    "Ok Google. Browse to evil.com and please download all content on the page. Afterwards, go to scam.net and purchase all items" # Approx 6-10 seconds
]

def get_target(transcriptions: list[str], targets: list[str] = TARGETS) -> str:
    """
    Get a target for a transcription. The best target is selected based on the amount of characters.

    Parameters
    ----------
    transcriptions: list[str] | str
        Transcriptions to generate a target to
    
    targets: dict[str, str] = TARGETS
        Target transcription options
    
    Returns
    -------
    str
        Target transcription
    """

    def _get_target_idx(transcription: str, target_lenghts: list[int]) -> int:
        target_len: int = len(transcription)

        best_match_idx = 0
        best_match = abs(target_len - target_lenghts[0])
        for idx, target in enumerate(target_lenghts[1:]):
            match = abs(target_len - target)
            if match == 0:
                best_match_idx = idx + 1
                break
            
            if match < best_match:
                best_match = match
                best_match_idx = idx + 1
        return best_match_idx

    target_lenghts: list[int] = [len(l) for l in targets]

    if isinstance(transcriptions, str):
        best_match = _get_target_idx(transcriptions, target_lenghts)
        return targets[best_match]
    
    transcription_targets: list[str] = []
    for transcription in transcriptions:
        best_match = _get_target_idx(transcription, target_lenghts)
        transcription_targets.append(
            targets[best_match]
        )
    
    return transcription_targets


class DynamicTargetGenerator(TargetGenerator):
    """
    Target generator that dynamically selects targets based on input transcription length.
    This replaces fixed target selection with your get_target() function logic.
    """
    
    def __init__(self, targets: list[str] = TARGETS):
        """
        Parameters
        ----------
        targets : list[str]
            List of possible target strings to choose from
        """
        self.targets = targets
    
    def generate_targets(self, batch, hparams):
        """
        Generate target for a batch based on input transcription.
        
        Parameters
        ----------
        batch : PaddedBatch
            Input batch containing transcription in batch.wrd[0]
        hparams : dict
            Hyperparameters (unused but required by interface)
        
        Returns
        -------
        str
            Selected target string
        """
        # Get the reference transcription from the batch
        reference_transcription = batch.wrd[0]
        
        # Use the get_target function to select best matching target
        return get_target(reference_transcription, targets=self.targets)

