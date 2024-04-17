from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "some example text in the English language"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy

print(output.shape)

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.float().numpy()[0])