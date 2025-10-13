import downloadWithDatalad
#import deleteWithDatalad
# import firstLevelModel
# import firstLevelModel_vices
import secondLevelModel_1vB
import secondLevelModel_1v6
#import secondLevelModel_1vB_vices
#import secondLevelModel_1v6_vices
# import secondLevelModel
#import transcribeAudio_x

stories = ['shapessocial','shapesphysical','21styear','bronx','pieman','piemanpni','tunnel']

task = "21styear"

#transcribeAudio_x.run(task)

# downloadWithDatalad.run(task)
# firstLevelModel.run(task)
# firstLevelModel_vices.run(task)
# secondLevelModel.run(task)
for x in range(1,8):
    secondLevelModel_1v6.run(task, x)
    secondLevelModel_1vB.run(task, x)
    #secondLevelModel_1v6_vices.run(task, x)
    #secondLevelModel_1vB_vices.run(task, x)

#deleteWithDatalad.run(task)