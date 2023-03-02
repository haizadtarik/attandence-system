from deepface import DeepFace
import cv2
from paddleocr import PaddleOCR
import re

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_ic(img_path):
    """
    This function detects IC number in an image and returns the IC number.

    Parameters:
    img_path (str): Path to the image file.

    Returns:
    str: IC number.

    Example:
    >>> ic = detect_ic("img.jpg")
    """
    pattern = re.compile(r'\d{6}-\d{2}-\d{4}')
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            if pattern.match(line[1][0]):
                return line[1][0]


def face_detection(img_path, detector_backend, target_size = (224, 224)):
    """
    This function detects faces in an image and returns a list of face objects using DeepFace.

    Parameters:
    img_path (str): Path to the image file.
    detector_backend (str): Backend to use for face detection. Options are 'opencv', 'ssd', 'dlib' and 'mtcnn'.

    Returns:
    list: List of face objects.

    Example:
    >>> face_objs = face_detection("img.jpg", target_size = (224, 224), 'ssd')
    """
    faces = DeepFace.extract_faces(img_path, target_size = target_size, detector_backend = detector_backend) 
    cv2.imwrite('db/'+img_path.split('/')[-1], 255*faces[0]['face'])


def face_recognition(img_path, db_path, detector_backend):
    """
    This function identify faces in an image by comparing it with faces in a database and returns 

    Parameters:
    img_path (str): Path to the image file.
    db_path (str): Path to the database folder.
    detector_backend (str): Backend to use for face detection. Options are 'opencv', 'ssd', 'dlib' and 'mtcnn'.

    Returns:
    list: List of face objects.

    Example:
    >>> face_objs = face_detection("img.jpg", 'ssd', target_size = (224, 224))
    """
    df = DeepFace.find(img_path, db_path, detector_backend)[0]
    if df.empty:
        return "Unknown"
    else:
        user_name = df[0]['identity'].values[0].split('/')[-1].split('.')[0]
        return user_name
        
        

if __name__ == "__main__":
    # faces = face_detection("jad.jpeg", 'ssd')
    identified_user = face_recognition("ic.jpeg", 'db', 'ArcFace')
    ic_num = detect_ic("ic.jpeg")
    print(f"User: {identified_user}, IC: {ic_num}")





