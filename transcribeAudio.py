import whisper
import whisper_timestamped as wt
import pandas as pd

#
#   Use PYTHON 12, ffmpeg needs to be installed, latest everything else
#
#   This transcribes audio files in using WHISPER
#

stories = ['tunnel','21styear','bronx','pieman','piemanpni']
#stories = ["bronx"]

m = whisper.load_model("large-v3")
transcript_text = ""

for task in stories:
    print( "Speech to text from: ./audio/" + task + "_audio.wav")
    result = wt.transcribe_timestamped(m, "audio/"+task+"_audio.wav", language="en", vad=True, verbose=True, temperature=0, condition_on_previous_text=False)
    df_words = pd.DataFrame(columns=["phrase","word","start", "end"])
    df_phrases = pd.DataFrame(columns=["text","start", "end"])
    phraseNumber = 0
    for segment in result['segments']:
        print( segment )
        phraseNumber = phraseNumber + 1
        df_phrases.loc[phraseNumber] = segment
        for word in segment['words']:
            df_words.loc[len(df_words)] = [phraseNumber, word['text'], word['start'], word['end']]
            transcript_text = transcript_text + " " + word['text']
    df_words.to_csv("./word_timestamps/"+task+"_transcription_per_word.csv", index=False)
    df_phrases.to_csv("./word_timestamps/"+task+"_transcription_per_phrase.csv", index=False)
    with open("./word_timestamps/"+task+"_transcription.txt", "w+") as fh:
        fh.write(transcript_text)