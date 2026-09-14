from pathlib import Path

from phase6_simulated_data.simulated_witness import (
    load_face_image,
    compute_face_encoding,
    select_best_candidate_with_confidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_IMAGE_DIR = PROJECT_ROOT / "data" / "starter_set" / "images"


def main():
    print("=" * 60)
    print("PHASE 6.4.2 — WITNESS CONFIDENCE TEST")
    print("=" * 60)

    target_path = STARTER_IMAGE_DIR / "starter_01.png"

    candidate_paths = [
        STARTER_IMAGE_DIR / f"starter_{i:02d}.png"
        for i in range(2, 13)
    ]

    print(f"Target: {target_path.name}")
    print(f"Candidates: {len(candidate_paths)}")

    # Load target
    target_image = load_face_image(target_path)
    target_encoding = compute_face_encoding(target_image)

    # Load candidate encodings
    candidate_encodings = []

    for path in candidate_paths:
        image = load_face_image(path)
        encoding = compute_face_encoding(image)
        candidate_encodings.append(encoding)

    # Simulate witness selection
    selected_index, scores, confidence = (
        select_best_candidate_with_confidence(
            target_encoding,
            candidate_encodings,
        )
    )

    print()
    print("Similarity scores:")

    for i, score in enumerate(scores):
        print(
            f"  {candidate_paths[i].name}: "
            f"{score:.4f}"
        )

    print()
    print(
        f"Selected candidate: "
        f"{candidate_paths[selected_index].name}"
    )

    print(
        f"Selected index: {selected_index}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # Validation
    assert len(scores) == 11
    assert 0 <= selected_index < 11
    assert 0.0 <= confidence <= 1.0
    assert confidence == scores[selected_index]

    print()
    print("=" * 60)
    print("PHASE 6.4.2 TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()