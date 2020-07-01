from GaitCore.Core import Point
frames = {}
frames["stairA"] = [Point.Point(0, 0, 0),
                    Point.Point(63, 0, 0),
                    Point.Point(0, 42, 0),
                    Point.Point(63, 49, 0)]

frames["stairB"] = [Point.Point(0, 0, 0),
                    Point.Point(49, 0, 0),
                    Point.Point(28, 56, 0),
                    Point.Point(70, 70, 0)]

file13 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_00.csv"
file12 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_02.csv"
file11 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig2_00.csv"
file10 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 stairclimbing_config1_01.csv"
file09 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_09/subject_09 stairclimbing_config1_00.csv"
file07 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 stairclimbing_config1_00.csv"
file06 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 stairclimbing_config1_02.csv"
file05 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_stair_config1_01.csv"
file03 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_stair_config0_02.csv"
file02 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_01.csv"
file01 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_03.csv"
file00 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig1_00.csv"



files = [ file00, file01, file02, file03, file05, file06, file07, file09]
sides = [ "R"] *len(files)