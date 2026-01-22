import string
import easyocr
import re
import cv2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
VN_PLATE_REGEX = re.compile(
    r"^\d{2}[A-Z]{1,2}\d{3}\d{2}$"
)


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text: str) -> bool:
    text = re.sub(r'[^A-Z0-9]', '', text)
    return VN_PLATE_REGEX.match(text) is not None

def format_license(text: str):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 7:
        return text

    chars = list(text)

    char_to_digit = {
        'O': '0',
        'I': '1', 
        'S': '5', 
        'Z': '2',
        'G': '9'}
    for i in [0, 1]:
        if chars[i].isalpha():
            chars[i] = char_to_digit.get(chars[i], chars[i])

    digit_to_char = {
        '0': 'O',
        '1': 'A',
        '2': 'Z',
        '4': 'A', 
        '5': 'S',
        '8': 'B'
    }
    if chars[2].isdigit():
        chars[2] = digit_to_char.get(chars[2], chars[2])
    
    newtext = ''.join(chars)

    print(f"new text: {newtext}")
    return newtext

import re

def normalize_vn_plate(text: str) -> str:
    text = text.upper().replace(" ", "")

    if "-" in text and "." in text:
        left, right = text.split("-", 1)

        right = re.sub(r'[^0-9]', '', right)

        if len(right) >= 5:
            return f"{left}-{right[:3]}.{right[3:5]}"
        return text

    raw = re.sub(r'[^A-Z0-9]', '', text)

    if len(raw) < 8:
        return text

    part1 = raw[:3]    
    part2 = raw[3:6]     
    part3 = raw[6:8]     

    char_to_digit = {
        'O': '0',
        'I': '1',
        'S': '5',
        'Z': '2',
        'B': '8',
        'G': '6',
        'G': '9'
    }

    part2 = ''.join(char_to_digit.get(c, c) for c in part2)
    part3 = ''.join(char_to_digit.get(c, c) for c in part3)

    if not (part2.isdigit() and part3.isdigit()):
        return text

    return f"{part1}-{part2}.{part3}"



def read_license_plate(license_plate_crop):
    #cv2.imwrite("debug_crop.jpg", crop)
    cv2.imwrite("debug_thresh.jpg", license_plate_crop)
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        print(f"Raw OCR : {text}")

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        newText = format_license(text)
        normalizeVnPlate = normalize_vn_plate(newText)
        if license_complies_format(normalizeVnPlate):
            print(f"Procesced image : {normalizeVnPlate}")
            return normalizeVnPlate, score
        else:
            print("Khong phai bien so")

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
