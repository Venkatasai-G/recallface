from pathlib import Path

import numpy as np
import face_recognition


def load_face_image(image_path: str | Path) -> np.ndarray:
    """
    Load an RGB face image for face_recognition.

    Args:
        image_path:
            Path to the image.

    Returns:
        RGB image as a NumPy array.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = face_recognition.load_image_file(
        str(image_path)
    )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape (H, W, 3)"
        )

    return image


def compute_face_encoding(
    image: np.ndarray,
) -> np.ndarray:
    """
    Compute a face_recognition encoding for one image.

    Returns:
        128-dimensional face encoding.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a NumPy array"
        )
    
    image = np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )

    locations = face_recognition.face_locations(image)

    if len(locations) == 0:
        raise ValueError(
            "No face detected in the image"
        )

    # Use the first detected face.
    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=[locations[0]],
        num_jitters=0,
    )

    if len(encodings) == 0:
        raise ValueError(
            "Could not compute a face encoding"
        )

    return np.asarray(encodings[0], dtype=np.float32)


def compute_similarity(
    target_encoding: np.ndarray,
    candidate_encoding: np.ndarray,
) -> float:
    """
    Convert face-recognition distance into a similarity score.

    Lower face distance means greater similarity.

    Similarity is defined as:

        similarity = 1 / (1 + distance)

    Therefore:
        distance >= 0
        similarity is in (0, 1].
    """

    target_encoding = np.asarray(
        target_encoding,
        dtype=np.float32,
    )

    candidate_encoding = np.asarray(
        candidate_encoding,
        dtype=np.float32,
    )

    if target_encoding.shape != (128,):
        raise ValueError(
            "target_encoding must have shape (128,)"
        )

    if candidate_encoding.shape != (128,):
        raise ValueError(
            "candidate_encoding must have shape (128,)"
        )

    distance = face_recognition.face_distance(
        [target_encoding],
        candidate_encoding,
    )[0]

    similarity = 1.0 / (1.0 + float(distance))

    return similarity


def select_best_candidate(
    target_encoding: np.ndarray,
    candidate_encodings: list[np.ndarray],
) -> tuple[int, list[float]]:
    """
    Simulate a witness selecting the candidate most similar
    to the target.

    Args:
        target_encoding:
            128-dimensional target face encoding.

        candidate_encodings:
            List of candidate face encodings.

    Returns:
        selected_index:
            Index of the candidate with highest similarity.

        similarity_scores:
            Similarity score for every candidate.
    """

    if len(candidate_encodings) == 0:
        raise ValueError(
            "At least one candidate encoding is required"
        )

    similarity_scores = [
        compute_similarity(
            target_encoding,
            candidate_encoding,
        )
        for candidate_encoding in candidate_encodings
    ]

    selected_index = int(
        np.argmax(similarity_scores)
    )

    return selected_index, similarity_scores

def select_best_candidate_with_confidence(
    target_encoding: np.ndarray,
    candidate_encodings: list[np.ndarray],
) -> tuple[int, list[float], float]:
    """
    Simulate witness selection and derive a confidence value.

    The candidate with the highest face-recognition similarity
    is selected.

    Confidence is the selected candidate's similarity score.
    """

    selected_index, similarity_scores = select_best_candidate(
        target_encoding,
        candidate_encodings,
    )

    confidence = float(
        similarity_scores[selected_index]
    )

    return (
        selected_index,
        similarity_scores,
        confidence,
    )