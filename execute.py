# import downloadWithDatalad
# #import deleteWithDatalad
# import firstLevelModel_virtues
# import firstLevelModel_vices
import secondLevelModel_1vB_MAC_virtues
import secondLevelModel_1v6_MAC_virtues
import secondLevelModel_1vB_MAC_vices
import secondLevelModel_1v6_MAC_vices
# import secondLevelModel_virtues
# import secondLevelModel_vices
# import transcribeAudio_x

#stories = ['shapessocial','shapesphysical','21styear','bronx','pieman','piemanpni','tunnel']
# stories = ['pieman', 'piemanpni']
#task = "pieman"

# for task in stories:
#     # transcribeAudio_x.run(task)
#     #
#     # downloadWithDatalad.run(task)
#     firstLevelModel_virtues.run(task)
#     firstLevelModel_vices.run(task)
#     secondLevelModel_virtues.run(task)
#     secondLevelModel_vices.run(task)
#
#
#     for foundation in range(1,8):
#         secondLevelModel_1v6_MAC_virtues.run(task, foundation)
#         secondLevelModel_1vB_MAC_virtues.run(task, foundation)
#         secondLevelModel_1v6_MAC_vices.run(task, foundation)
#         secondLevelModel_1vB_MAC_vices.run(task, foundation)

    #deleteWithDatalad.run(task)

for foundation in range(1,8):
    secondLevelModel_1v6_MAC_virtues.run("bronx", foundation)
    secondLevelModel_1vB_MAC_virtues.run("bronx", foundation)
    secondLevelModel_1v6_MAC_vices.run("bronx", foundation)
    secondLevelModel_1vB_MAC_vices.run("bronx", foundation)
