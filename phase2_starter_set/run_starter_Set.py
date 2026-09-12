import sys
from pathlib import Path

import torch

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    STARTER_SET_DIR,
    STARTER_SET_SIZE,
    STARTER_POOL_MAX,
    RANDOM_SEED,
)

from phase1_diffae.load_model import load_diffae
from phase1_diffae.inference import generate_face

from phase2_starter_set.build_starter_set import (
    sample_latent_pool,
    select_representative_latents,
    create_fixed_xT,
    save_latents,
    save_xT,
    save_starter_images,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "last.ckpt"
)

LATENT_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "latent.pkl"
)

POOL_SIZE = STARTER_POOL_MAX
SEED = RANDOM_SEED


# ============================================================
# Main
# ============================================================

def main():

    print("============================================")
    print("RecallFace Phase 2")
    print("Round 1 Starter Set")
    print("============================================")

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Latent pool:", LATENT_PATH)
    print("Pool size:", POOL_SIZE)
    print("Starter size:", STARTER_SET_SIZE)
    print("Seed:", SEED)

    # --------------------------------------------------------
    # Load pretrained DiffAE
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 1: Loading pretrained DiffAE")
    print("============================================")

    model, conf, device = load_diffae(
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Load learned latent distribution
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 2: Loading learned latent distribution")
    print("============================================")

    latent_data = torch.load(
        LATENT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    conds = latent_data["conds"].float()

    print("Learned latent pool:", tuple(conds.shape))

    # --------------------------------------------------------
    # Sample 500–1000 latent codes
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 3: Sampling latent candidate pool")
    print("============================================")

    latent_pool = sample_latent_pool(
        conds=conds,
        pool_size=POOL_SIZE,
        seed=SEED,
    )

    print(
        "Sampled latent pool:",
        tuple(latent_pool.shape)
    )

    # --------------------------------------------------------
    # KMeans selection
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 4: Selecting diverse starter latents")
    print("============================================")

    starter_latents = select_representative_latents(
        latent_pool=latent_pool,
        starter_size=STARTER_SET_SIZE,
        seed=SEED,
    )

    print(
        "Selected starter latents:",
        tuple(starter_latents.shape)
    )

    # --------------------------------------------------------
    # Create ONE fixed xT
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 5: Creating fixed stochastic xT")
    print("============================================")

    fixed_xT = create_fixed_xT(
        starter_size=STARTER_SET_SIZE,
        image_size=conf.img_size,
        seed=SEED,
    )

    print("Fixed xT shape:", tuple(fixed_xT.shape))

    # --------------------------------------------------------
    # Generate 12 starter faces
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 6: Generating starter faces")
    print("============================================")

    starter_latents_device = starter_latents.to(device)
    fixed_xT_device = fixed_xT.to(device)

    images = generate_face(
        model=model,
        conf=conf,
        device=device,
        z=starter_latents_device,
        x_T=fixed_xT_device,
        T=conf.T_eval,
    )

    print(
        "Generated images:",
        tuple(images.shape)
    )

    # --------------------------------------------------------
    # Save permanent starter set
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Step 7: Saving permanent starter set")
    print("============================================")

    STARTER_SET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    latent_output = (
        STARTER_SET_DIR
        / "starter_latents.pt"
    )

    xT_output = (
        STARTER_SET_DIR
        / "starter_xT.pt"
    )

    images_output = (
        STARTER_SET_DIR
        / "images"
    )

    save_latents(
        starter_latents,
        latent_output,
    )

    save_xT(
        fixed_xT,
        xT_output,
    )

    save_starter_images(
        images,
        images_output,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Phase 2 Starter Set Complete")
    print("============================================")

    print("Latents:", latent_output)
    print("xT:", xT_output)
    print("Images:", images_output)

    print()
    print("Saved images:")

    for image_path in sorted(images_output.glob("*.png")):
        print(
            " -",
            image_path.name,
            f"({image_path.stat().st_size / 1024:.1f} KB)"
        )

    print()
    print("============================================")
    print("12 permanent starter faces created.")
    print("============================================")


if __name__ == "__main__":
    main()