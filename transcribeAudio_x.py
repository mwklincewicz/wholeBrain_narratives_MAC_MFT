import whisperx
import gc
import os
import pandas as pd

#
#   Use PYTHON 12, ffmpeg needs to be installed, latest everything else
#
#   This transcribes audio files in using WHISPERX, which builds on OpenAI whisper model for speech-to-text
#

def run(task):
    device = "cpu"
    batch_size = 4  # reduce if low on GPU mem
    compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    transcript_text = ""

    audio_file = ".\\data\\audio\\" + task + "_audio.wav"
    if os.path.exists(audio_file):
        print("The audio file " + audio_file + " exists.")
    else:
        print("The audio file " + audio_file + " DOES NOT EXIST.")
    print( "Speech to text from: ./data/audio/" + task + "_audio.wav")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en", batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    df_words = pd.DataFrame(columns=["phrase","word","start", "end"])
    df_phrases = pd.DataFrame(columns=["text","start", "end"])
    phraseNumber = 0
    for segment in result['segments']:
        print( segment )
        phraseNumber = phraseNumber + 1
        df_phrases.loc[phraseNumber] = segment
        for word in segment['words']:
            df_words.loc[len(df_words)] = [phraseNumber, word['word'], word['start'], word['end']]
            transcript_text = transcript_text + " " + word['word']
    os.makedirs("./text/timestamps/"+task+"/", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    df_words.to_csv("./text/timestamps/"+task+"/"+task+"_transcription_per_word_x.csv", index=False)
    df_phrases.to_csv("./text/timestamps/"+task+"/"+task+"_transcription_per_phrase_x.csv", index=False)
    with open("./text/timestamps/"+task+"/"+task+"_transcription_x.txt", "w+") as fh:
        fh.write(transcript_text)

    import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a
