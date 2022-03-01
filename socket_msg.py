'''signal to functionality mapping'''

requests_dic = {
    "00": 'Match_Start_Image_Xml',
    '01': ":Match_End_Xml",
    "20": 'Track_Start_New_Image_Xml',
    "21": "Track_Start_Image_Left",
    "22": 'Track_Start_Image_Right',
    "23": 'Track_End_Xml',
    "40": 'Mask_Start_Image_Xml',
    "41": 'Mask_End',
    "42": 'Mask_Respond_Image',
    "43": 'Mask_Resquest_Image'
}

file_path_dic = {
    'Path_Picture_Left': "./Algorithm/Pic/left.jpg",
    'Path_Picture_Right': "./Algorithm/Pic/right.jpg",
    'Path_Xml_Match': "./Algorithm/Xml/Match/match.xml",
    'Path_Xml_Track_Left': "./Algorithm/Xml/Track/trackLeft.xml",
    'Path_Xml_Track_Right': "./Algorithm/Xml/Track/trackRight.xml",
    'Path_Xml_Mask_Left': "./Algorithm/Xml/Mask/maskLeft.xml",
    'Path_Xml_Mask_Right': "./Algorithm/Xml/Mask/maskRight.xml",
}

xml_coordinates_dicimal_precision = 3
