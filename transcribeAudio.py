import whisper
import whisper_timestamped as wt
import pandas as pd

stories = ['tunnel','21styear','bronx','pieman','piemanpni']

m = whisper.load_model("large-v3")

for task in stories:
    print( "Speech to text from: ./audio/" + task + "_audio.wav")
    result = wt.transcribe_timestamped(m, "audio/"+task+"_audio.wav", language="en", vad=True, verbose=True, temperature=0, condition_on_previous_text=False)
    df = pd.DataFrame(columns=["phrase","word","start", "end"])

    phraseNumber = 0
    for segment in result['segments']:
        phraseNumber = phraseNumber + 1
        for word in segment['words']:
            df.loc[len(df)] = [phraseNumber, word['text'], word['start'], word['end']]
    df.to_csv("./word_timestamps/"+task+"_transcription.csv", index=False)