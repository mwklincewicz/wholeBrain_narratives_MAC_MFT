#import downloadWithDatalad
#import deleteWithDatalad
import firstLevelModel
#import secondLevelModel
#import transcribeAudio_x

stories = ['shapessocial','shapesphysical']#,'21styear','bronx','pieman','piemanpni','tunnel']
task = "bronx"

#transcribeAudio_x.run(task)
#downloadWithDatalad.run(task)
firstLevelModel.run(task)
#secondLevelModel.run(task)
#deleteWithDatalad.run(task)

