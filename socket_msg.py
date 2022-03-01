"""signal to functionality mapping"""

signal_2_requests_dic = {
    "00": 'Match_Start_Image_Xml',
    '01': "Match_End_Xml",

    "20": 'Track_Start_New_Image_Xml_Left',
    "21": "Track_Start_Image_Left",
    "23": 'Track_End_Xml_Left',

    "31": 'Track_Start_New_Image_Xml_Right',
    "32": 'Track_Start_Image_Right',
    "33": 'Track_End_Xml_Right',

    "40": 'Mask_Start_Image_Xml_Left',
    "41": 'Mask_End_Left',
    "42": 'Mask_Respond_Image_Left',
    "43": 'Mask_Resquest_Image_Left',

    "50": 'Mask_Start_Image_Xml_Right',
    "51": 'Mask_End_Right',
    "52": 'Mask_Respond_Image_Right',
    "53": 'Mask_Resquest_Image_Right'
}

requests_2_signal_dic = {}
for key in signal_2_requests_dic.keys():
    value = signal_2_requests_dic[key]
    requests_2_signal_dic[value] = key


file_path_dic = {
    'Path_Picture_Left': "./Algorithm/Pic/left.jpg",
    'Path_Picture_Right': "./Algorithm/Pic/right.jpg",
    'Path_Xml_Match_Left': "./Algorithm/Xml/Match/matchLeft.xml",
    'Path_Xml_Match_Right': "./Algorithm/Xml/Match/matchRight.xml",
    'Path_Xml_Match_Id': "./Algorithm/Xml/Match/matchId.xml",

    'Path_Xml_Track_Left': "./Algorithm/Xml/Track/trackLeft.xml",
    'Path_Xml_Track_Right': "./Algorithm/Xml/Track/trackRight.xml",
    'Path_Xml_Mask_Left': "./Algorithm/Xml/Mask/maskLeft.xml",
    'Path_Xml_Mask_Right': "./Algorithm/Xml/Mask/maskRight.xml",
}

xml_coordinates_dicimal_precision = 3
