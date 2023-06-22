#%%
from pathlib import Path
import numpy as np
from mtrf.model import TRF

root = Path(__file__).parent.absolute()

speech_response = np.load(root / "data" / "speech_data.npy", allow_pickle=True).item()
fs = speech_response["samplerate"][0][0]
response = speech_response["response"]
stimulus = speech_response["stimulus"]
# %%
trf_encoder = TRF()
tmin, tmax = -0.1, 0.2
trf_encoder.train(stimulus, response, fs, tmin, tmax, 10)
# use the trained TRF to predict data
prediction2, correlation2, error2 = trf_encoder.predict(
    stimulus, response, average=False
)
# %%
error2
# %%
correlation2
# %%
