import torch

from phase11_integration.navigation_engine import NavigationEngine


def main():
    print("=" * 60)
    print("PHASE 11 INTEGRATION TEST")
    print("=" * 60)

    print("\nLoading NavigationEngine...")

    engine = NavigationEngine(
        device="cuda",
        num_candidates=12,
        latent_clip=3.0,
    )

    print("\nNavigationEngine loaded successfully.")
    print("Device:", engine.device)
    print("Number of candidates:", engine.num_candidates)

    # Use one random 512-D latent only for integration testing.
    current_latent = torch.randn(
        512,
        dtype=torch.float32,
    )

    print("\nCurrent latent shape:", tuple(current_latent.shape))

    print("\nGenerating Round 2 candidates...")

    candidate_latents, candidate_images = (
        engine.generate_round(
            current_latent=current_latent,
            round_number=2,
        )
    )

    print("\nCandidate generation successful.")
    print(
        "Candidate latent shape:",
        tuple(candidate_latents.shape),
    )

    print(
        "Candidate image array shape:",
        candidate_images.shape,
    )

    print(
        "Candidate image dtype:",
        candidate_images.dtype,
    )

    print(
        "Candidate latent finite:",
        torch.isfinite(candidate_latents).all().item(),
    )

    print(
        "Candidate image count:",
        len(candidate_images),
    )

    print(
        "Candidate image size:",
        candidate_images.shape[1:],
    )

    assert candidate_latents.shape == (12, 512)
    assert len(candidate_images) == 12
    assert candidate_images.shape[1:] == (256, 256, 3)
    assert torch.isfinite(candidate_latents).all()

    print("\n" + "=" * 60)
    print("PHASE 11 INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()