import whisper
import whisper_timestamped as wt
import pandas as pd

task = 'tunnel'
m = whisper.load_model("large-v3")

result = wt.transcribe_timestamped(m, "audio/"+task+"_audio.wav", language="en", vad=True, verbose=True, temperature=0, condition_on_previous_text=False)

df = pd.DataFrame(columns=["phrase","word","start", "end"])

phraseNumber = 0
for segment in result['segments']:
    phraseNumber = phraseNumber + 1
    for word in segment['words']:
        df.loc[len(df)] = [phraseNumber, word['text'], word['start'], word['end']]
df.to_csv("./word_timestamps/"+task+"transcription.csv", index=False)