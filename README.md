# MORALITY IN THE BRAIN - NARRATIVES

### I.      Find neural correlates of foundation scores through whole-brain analysis MAC family foundation peaks

### II.     Peaks for other foundations from narratives < .1 in any

### III.    Do per segment and per sentence analysis for all foundations for each story, showing whole brain activations

............................................................................................
#### **The pipeline for analysis is the following:**

1) Get audio files from repo on Drive or Ken Normal lab files (put in .\data\audio\, if not there already)
2) Obtain a transcript (put it in .\text\text_to_be_segmented) or use the one generated via transcribeAudio_x.py (in .\text\timestamps)
3) Manually, with ChatGPT, or some other means, generate segment .txt files and put them into .\text\segments
4) Run transcribeAudio_x.py to obtain timestamp .csv files (per word/phrase) in .\text\timestamps
5) Run getMAC_MFT_for_segments.py to obtain MAC/MFT, all/per-word, vice/virtue, scores per sentence, per segment (from 3), and durations per sentence (from (4))
6) Download .nii and .tsv files for a story using downloadWithDatalad.py into .\data\allDataAliases\fmriprep (will have a prompt for task name)
7) Run firstLevelModel_XXXXXXXXXX.py, as for task, etc. TODO: make these generic
8) Run secondLevelModel_XXXXXXXXX.py, as for task, etc. TODO: make these generic
9) All processed data and plots will be stored in drive G: on WHITE_LADY, which syncs with Google Drive.  
10) Delete .nii and .tsv files for a story using deleteWithDatalad.py (will have a prompt for task name)

............................................................................................
#### **Some notable ways to break things:**

a) MAC and MFT scoring in (5) above uses libraries that have dependencies on outdated versions of spacy, typing extensions, etc., so downgrade to Python 3.8 and follow instructions from comments in getMAC_MFT_for_segments.py to complete step (5) ONLY

b) (4) depends on ffmpeg.  Follow online instructions to install it or you will get a mysterious File missing error

c) Other types of analysis, Jupyter Notebooks, etc., all have their own directories, so pay attention when executing anything that you are in the right directory

d) .\results, .\testData, are not used, but can be for debugging and not breaking things, so use them!

e) .\data\audio, .\data\allDataAliases have datafiles from Princeton (Ken Norman Lab), which are both large, and not yours. They are in .gitignore, so they don't get pushed with commits

f) my_temp_file and emfdTemp.csv are recreated every time you do (5) and they are there to easily and slowly handle character encoding issues; they sometimes tick around and this is a pain right now and slows things down, so it needs a better method.

g) .\text\Narratives_participants.csv holds information about the order in which participants heard social and physical shapes stories

h) .\text\emacscore-master.zip needs to be used to install MAC dictionary scoring for (5), keep it in the directory and use pip

i) drive G: on WHITE_LADY, which syncs with Google Drive, sometimes does not have a check for directories; manually make them.  