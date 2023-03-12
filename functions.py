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
    detector_backend (str): Backend to use for face detection. Options are 'opencv', 'retinaface', 'ssd', 'dlib' and 'mtcnn'.

    Returns:
    list: List of face objects.

    Example:
    >>> face_objs = face_detection("img.jpg", target_size = (224, 224), 'ssd')
    """
    faces = DeepFace.extract_faces(img_path, target_size = target_size, detector_backend = detector_backend) 
    return faces


def face_verification(faces, model_name='Facenet', similarity_metric = 'cosine'):
    """
    This function verify whether faces are the same person.

    Parameters:
    faces (list): List of face objects.
    model_name (str): model used for verification. Options are Facenet, VGG-Face, ArcFace and Dlib.
    similarity_metric (string): cosine, euclidean, euclidean_l2

    Returns:
    str: same or different.

    Example:
    >>> result = face_verification(faces)
    >>> result = face_verification(faces, model_name = 'ArcFace')
    """
    if len(faces) > 1:
        face_width = [face['facial_area']['w'] for face in faces]
        user_face_index = face_width.index(max(face_width))
        ic_face_index = face_width.index(sorted(face_width)[-2])
        user_face = faces[user_face_index]['face']
        ic_face = faces[ic_face_index]['face']
        result = DeepFace.verify(user_face, ic_face, model_name = model_name, distance_metric=similarity_metric, enforce_detection = False)
        return result['verified']
    else:
        return "Not enough face detected"
        
def verify_attendence(img_path, face_detector='retinaface', target_size = (224, 224), model_name='Facenet', similarity_metric = 'cosine'):
    """
    This verify whether the person holding the IC is the same person as the IC owner.

    Parameters:
    img_path (str): Path to the image file.
    face_detector (str): Backend to use for face detection. Options are 'opencv', 'retinaface', 'ssd', 'dlib' and 'mtcnn'.
    target_size (tuple): Size of the face image to be extracted.
    model_name (str): model used for verification. Options are Facenet, VGG-Face, ArcFace and Dlib.
    similarity_metric (string): cosine, euclidean, euclidean_l2

    Returns:
    str: IC number if same person error if different person

    Example:
    >>> result = verify_attendence("img.jpg")
    """
    faces = face_detection(img_path, face_detector, target_size = target_size)
    result = face_verification(faces, model_name = model_name, similarity_metric = similarity_metric)
    if type(result) != str:
        if result:
            ic_num = detect_ic(img_path)
            if ic_num:
                return f"IC: {ic_num}"
            else:
                return "IC not detected"
        else:
            return "Different user holding IC"
    else:
        return result

if __name__ == "__main__":
    result = verify_attendence("jad.jpeg")
    print(result)
    




