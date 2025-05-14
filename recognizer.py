import cv2
import numpy as np
import pickle
import uuid
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Configuration
REQUIRED_SIZE = (160, 160)
RECOGNITION_THRESHOLD = 0.8

# Initialize once
facenet = FaceNet()
detector = MTCNN()

def initialize_database():
    """Create a new empty database structure."""
    return {"names": [], "embeddings": [], "uuids": [], "nims": []}

def load_database(path='database.pkl'):
    try:
        with open(path, 'rb') as f:
            database = pickle.load(f)

        # Convert embeddings to list if they are numpy arrays
        if isinstance(database['embeddings'], np.ndarray):
            database['embeddings'] = database['embeddings'].tolist()
            
        # Add uuids field if it doesn't exist (for backwards compatibility)
        if 'uuids' not in database:
            database['uuids'] = [str(uuid.uuid4()) for _ in database['names']]
            save_database(database, path)
            
        # Add nims field if it doesn't exist (for backwards compatibility)
        if 'nims' not in database:
            database['nims'] = ["" for _ in database['names']]
            save_database(database, path)

        print(f"✅ Loaded face database with {len(database['names'])} entries.")
        return database
    except FileNotFoundError:
        print("❌ Warning: 'database.pkl' not found. Creating a new empty database.")
        new_database = initialize_database()
        save_database(new_database, path)
        return new_database


def save_database(database, path='database.pkl'):
    # Convert embeddings back to NumPy array before saving
    database['embeddings'] = np.array(database['embeddings'])
    with open(path, 'wb') as f:
        pickle.dump(database, f)
    print(f"✅ Database saved with {len(database['names'])} entries.")


def preprocess_face(face, required_size=REQUIRED_SIZE):
    try:
        face = cv2.resize(face, required_size)
        return np.asarray(face)
    except:
        return None

def get_embedding(face_array):
    return facenet.embeddings([face_array])[0]

def recognize_face(embedding, database):
    if database is None or len(database['embeddings']) == 0:
        return "Unknown", 0.0

    distances = np.linalg.norm(np.array(database['embeddings']) - embedding, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    if min_distance < 0.8:
        name = database['names'][min_idx]
        confidence = min_distance
        return name, confidence
    else:
        return "Unknown", 0.0


def detect_faces_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return detector.detect_faces(rgb_frame), rgb_frame
