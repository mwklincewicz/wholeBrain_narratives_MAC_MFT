# MORALITY IN THE BRAIN - NARRATIVES

### I.      Find neural correlates of moral/non-moral detector through whole-brain analysis

### II.     Do per segment, per chunk, or per sentence analysis for all foundations for each story, showing whole brain activations

### III.    Do RSA analysis for foundations that involve the hippocampus

............................................................................................
#### **The pipeline for analysis is the following:**

1) get audio files into ./audio, then install datalad, git-annex, and run: datalad clone https://datasets.datalad.org/labs/hasson/narratives/derivatives/fmriprep
2) Run transcribe(NAME) to obtain timestamp .csv files (per word/phrase) in .\text\timestamps
3) Run getMAC_MFT_by_seconds.run(SECONDS) to obtain MAC/MFT, all, vice/virtue, scores per sentence and durations per chunk in SECONDS (from (2))
4) Download .nii and .tsv files for a story using downloadStory(NAME) into .\fmriprep
5) Run firstLevelXXXXX(NAME, DIRECTORY, SCORING), as for task, etc. NAME=story, DIRECTORY=where processed files will go, SCORING=seconds per chunk
6) Run secondLevelModel_XXXXXXXXX(NAME, DIRECTORY, SCORING), as for (5), etc.
7) All processed data and plots will be stored in DIRECTORY, which you should independently sync with Google Drive or something like that.  
8) univeriateWithMask takes NAME, DIRECTORY, SCORING, and MASK, which is a string name+extension of a nii.gz file in ./masks/
9) Drop fMRI .nii and confound regressor .tsv files for a story using dropStory(NAME); this does not delete the alias file!

............................................................................................
#### **Some notable ways to break things:**

a) MAC and MFT scoring uses libraries that have dependencies on outdated versions of spacy, typing extensions, etc., so downgrade to Python 3.8 and follow instructions from comments in getMAC_MFT to complete this step

b) (2) depends on ffmpeg.  Follow online instructions to install it and its own dependencies or you will get a mysterious File missing error

c) Other types of analysis, Jupyter Notebooks, etc., all have their own directories, so pay attention when executing anything that you are in the right directory

d) There are directories with backups and other analyses that are not used, but can be for debugging and not breaking things, so use them!

e) .\audio, .\fmriprep have datafiles from Princeton (Ken Norman Lab), which are both large, and not yours. They are in .gitignore, so they don't get pushed with commits

f) my_temp_file and emfdTemp.csv are recreated in root directory every time you do score with MFT/MAC and they are there to handle character encoding issues; they sometimes stick around, so this is a TODO.

g) .\text\Narratives_participants.csv holds information about the order in which participants heard social and physical shapes stories

h) .\text\emacscore-master.zip needs to be used to install MAC dictionary scoring; keep it in the directory and use pip

i) We typically use drive G: on WHITE_LADY, which syncs with Google Drive, sometimes does not have a check for directories; manually make them. 

j) excluded.xlsx contains per story ids for participants that should not be used in analysis for a story and this is how we define groups for a story, if there are any (excluded from a story are NOT in a group); update as needed
