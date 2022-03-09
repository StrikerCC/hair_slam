"""signal to functionality mapping"""

signal_2_requests_dic = {
    "00": 'Match_Start_Image_Xml',
    '01': "Match_End_Xml",

    "20": 'Track_Start_New_Image_Xml_Left',
    "21": "Track_Start_Image_Left",
    "22": 'Track_End_Xml_Left',
    "23": 'Push_Mask_Xml_Left',
    "24": 'Peek_Mask_Xml_Left',

    "30": 'Track_Start_New_Image_Xml_Right',
    "31": 'Track_Start_Image_Right',
    "32": 'Track_End_Xml_Right',
    "33": 'Push_Mask_Xml_Right',
    "34": 'Peek_Mask_Xml_Right',

    # "40": 'Mask_Start_Image_Xml_Left',
    # "41": 'Mask_End_Left',
    # "42": 'Mask_Request_Left',
    #
    # "50": 'Mask_Start_Image_Xml_Right',
    # "51": 'Mask_End_Right',
    # "52": 'Mask_Request_Right'
}

requests_2_signal_dic = {}
for key in signal_2_requests_dic.keys():
    value = signal_2_requests_dic[key]
    requests_2_signal_dic[value] = key


file_path_dic = {
    'Path_Picture_Match_Left':    "D:/Algorithm/Pic/matchLeft.jpg",
    'Path_Picture_Match_Right':    "D:/Algorithm/Pic/matchRight.jpg",

    'Path_Picture_Track_Left':   "D:/Algorithm/Pic/trackLeft.jpg",
    'Path_Picture_Track_Right':   "D:/Algorithm/Pic/trackRight.jpg",

    'Path_Xml_Match_Left':  "D:/Algorithm/Xml/Match/matchLeft.xml",
    'Path_Xml_Match_Right': "D:/Algorithm/Xml/Match/matchRight.xml",
    'Path_Xml_Match_Id':    "D:/Algorithm/Xml/Match/matchId.xml",

    'Path_Xml_Track_Left':  "D:/Algorithm/Xml/Track/trackLeft.xml",
    'Path_Xml_Tracked_Left': "D:/Algorithm/Xml/Track/trackedLeft.xml",

    'Path_Xml_Track_Right':         "D:/Algorithm/Xml/Track/trackRight.xml",
    'Path_Xml_Tracked_Right':       "D:/Algorithm/Xml/Track/trackedRight.xml",

    'Path_Xml_Mask_Push_Left':   "D:/Algorithm/Xml/Mask/maskLeft.xml",
    'Path_Xml_Mask_Tracked_Left':   "D:/Algorithm/Xml/Mask/maskTrackedLeft.xml",

    'Path_Xml_Mask_Push_Right':  "D:/Algorithm/Xml/Mask/maskRight.xml",
    'Path_Xml_Mask_Tracked_Right':  "D:/Algorithm/Xml/Mask/maskTrackedRight.xml",
}


# file_path_dic = {
#     'Path_Picture_Left':    "./Algorithm/Pic/left.jpg",
#     'Path_Picture_Right':   "./Algorithm/Pic/right.jpg",
#     'Path_Xml_Match_Left':  "./Algorithm/Xml/Match/matchLeft.xml",
#     'Path_Xml_Match_Right': "./Algorithm/Xml/Match/matchRight.xml",
#     'Path_Xml_Match_Id':    "./Algorithm/Xml/Match/matchId.xml",
#
#     'Path_Xml_Track_Left':  "D:/Algorithm/Xml/Track/trackLeft.xml",
#     'Path_Xml_Track_Right': "D:/Algorithm/Xml/Track/trackRight.xml",
#     'Path_Xml_Mask_Left':   "D:/Algorithm/Xml/Mask/maskLeft.xml",
#     'Path_Xml_Mask_Right':  "D:/Algorithm/Xml/Mask/maskRight.xml",
# }

xml_coordinates_decimal_precision = 3
