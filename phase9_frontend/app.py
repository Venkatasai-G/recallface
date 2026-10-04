from pathlib import Path
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RecallFace",
    page_icon="👤",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STARTER_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "starter_set"
    / "images"
)

MAX_ROUNDS = 5


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "session_started" not in st.session_state:
    st.session_state["session_started"] = False

if "selected_face" not in st.session_state:
    st.session_state["selected_face"] = None

if "selected_image_path" not in st.session_state:
    st.session_state["selected_image_path"] = None

# IMPORTANT:
# This stores the most recently selected image permanently,
# even after selected_image_path is reset for the next round.
if "latest_selected_image_path" not in st.session_state:
    st.session_state["latest_selected_image_path"] = None

if "current_round" not in st.session_state:
    st.session_state["current_round"] = 1

if "fine_tuning" not in st.session_state:
    st.session_state["fine_tuning"] = False

if "show_final_output" not in st.session_state:
    st.session_state["show_final_output"] = False

if "final_image_path" not in st.session_state:
    st.session_state["final_image_path"] = None

if "fine_tuning_values" not in st.session_state:
    st.session_state["fine_tuning_values"] = {}

if "fine_tuning_complete" not in st.session_state:
    st.session_state["fine_tuning_complete"] = False


current_round = st.session_state["current_round"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_session():
    """
    Completely reset the current RecallFace session.
    """

    keys_to_reset = [
        "session_started",
        "selected_face",
        "selected_image_path",
        "latest_selected_image_path",
        "current_round",
        "fine_tuning",
        "show_final_output",
        "final_image_path",
        "fine_tuning_values",
        "fine_tuning_complete",
    ]

    for key in keys_to_reset:
        if key == "session_started":
            st.session_state[key] = False

        elif key == "current_round":
            st.session_state[key] = 1

        elif key in [
            "selected_face",
            "selected_image_path",
            "latest_selected_image_path",
            "final_image_path",
        ]:
            st.session_state[key] = None

        elif key == "fine_tuning_values":
            st.session_state[key] = {}

        else:
            st.session_state[key] = False


def get_starter_images():
    """
    Return the Phase 2 starter images in deterministic order.
    """

    if not STARTER_IMAGE_DIR.exists():
        return []

    image_paths = sorted(
        list(STARTER_IMAGE_DIR.glob("*.png"))
        + list(STARTER_IMAGE_DIR.glob("*.jpg"))
        + list(STARTER_IMAGE_DIR.glob("*.jpeg"))
    )

    return image_paths


def get_round_images(round_number):
    """
    Phase 9 placeholder behavior.

    Round 1:
        Uses the actual Phase 2 starter set.

    Later rounds:
        Uses the same starter images in reverse order.

    Real Projector/Sampler generation will be connected
    during Phase 11.
    """

    starter_images = get_starter_images()

    if round_number == 1:
        return starter_images

    return list(reversed(starter_images))


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state["session_started"]:

    st.title("👤 RecallFace")

    st.subheader(
        "Interactive AI-Assisted Facial Composite Generation"
    )

    st.write(
        """
        RecallFace is an interactive facial composite system designed
        to help an eyewitness identify a face through recognition-based
        selection rather than relying only on verbal descriptions.
        """
    )

    st.markdown("### How it works")

    st.markdown(
        """
        1. Start a witness session.
        2. Select the face that looks most similar to the target.
        3. Provide your confidence level.
        4. Optionally provide textual guidance.
        5. Continue through multiple rounds.
        6. Review the session history.
        7. Fine-tune the final face if required.
        8. Generate and download the final composite.
        """
    )

    st.info(
        "Phase 9 currently uses simulated round generation. "
        "The trained Projector/Sampler and DiffAE will be integrated "
        "during Phase 11."
    )

    if st.button(
        "🚀 Start Session",
        type="primary",
        use_container_width=False,
    ):

        st.session_state["session_started"] = True
        st.session_state["current_round"] = 1
        st.session_state["selected_face"] = None
        st.session_state["selected_image_path"] = None
        st.session_state["latest_selected_image_path"] = None
        st.session_state["fine_tuning"] = False
        st.session_state["show_final_output"] = False
        st.session_state["final_image_path"] = None
        st.session_state["fine_tuning_values"] = {}
        st.session_state["fine_tuning_complete"] = False

        st.rerun()


# ============================================================
# FINE-TUNING VIEW
# ============================================================

elif st.session_state["fine_tuning"]:

    st.title("🎛️ Fine-tune Face")

    st.info(
        """
        The fine-tuning controls are currently simulated for Phase 9.
        In Phase 11, these controls will call the trained Projector
        directly on individual feature axes.
        """
    )

    st.markdown("### Adjust individual facial features")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Simulated age adjustment.",
        )

        hair = st.slider(
            "Hair",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Simulated hair adjustment.",
        )

        face_shape = st.slider(
            "Face Shape",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Simulated face shape adjustment.",
        )

    with col2:

        smile = st.slider(
            "Smile",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Simulated smile adjustment.",
        )

        eyes = st.slider(
            "Eyes",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="Simulated eye adjustment.",
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DISPLAY CURRENT SELECTED IMAGE
    # --------------------------------------------------------

    current_image = (
        st.session_state.get("latest_selected_image_path")
        or st.session_state.get("selected_image_path")
    )

    if current_image:

        current_image_path = Path(current_image)

        if current_image_path.exists():

            st.markdown("### Current Composite")

            st.image(
                str(current_image_path),
                caption="Current selected face",
                width=350,
            )

        else:

            st.warning(
                "The selected image could not be found."
            )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "Apply Fine-tuning",
            type="primary",
            use_container_width=True,
        ):

            st.session_state["fine_tuning_values"] = {
                "age": age,
                "hair": hair,
                "face_shape": face_shape,
                "smile": smile,
                "eyes": eyes,
            }

            st.success(
                "Fine-tuning values applied."
            )

    with col2:

        if st.button(
            "Finalize Face",
            use_container_width=True,
        ):

            # IMPORTANT:
            # Use latest_selected_image_path instead of
            # selected_image_path because selected_image_path
            # may have been reset after Continue.
            final_path = (
                st.session_state.get(
                    "latest_selected_image_path"
                )
                or st.session_state.get(
                    "selected_image_path"
                )
            )

            st.session_state["final_image_path"] = final_path
            st.session_state["fine_tuning_complete"] = True
            st.session_state["fine_tuning"] = False
            st.session_state["show_final_output"] = True

            st.rerun()

    with col3:

        if st.button(
            "← Back to Session Review",
            use_container_width=True,
        ):

            st.session_state["fine_tuning"] = False

            st.rerun()


# ============================================================
# FINAL OUTPUT PAGE
# ============================================================

elif st.session_state["show_final_output"]:

    st.title("Final Composite")

    st.success(
        "Your facial composite is ready."
    )

    final_image = st.session_state.get(
        "final_image_path"
    )

    # --------------------------------------------------------
    # DISPLAY FINAL IMAGE
    # --------------------------------------------------------

    if final_image:

        final_image_path = Path(final_image)

        if final_image_path.exists():

            st.markdown("### Final Face")

            st.image(
                str(final_image_path),
                caption="RecallFace Final Composite",
                width=500,
            )

            # ------------------------------------------------
            # DOWNLOAD PNG
            # ------------------------------------------------

            try:

                with open(
                    final_image_path,
                    "rb",
                ) as image_file:

                    image_bytes = image_file.read()

                st.download_button(
                    label="⬇️ Download PNG",
                    data=image_bytes,
                    file_name="recallface_final.png",
                    mime="image/png",
                    type="primary",
                )

            except Exception as e:

                st.error(
                    f"Unable to prepare the image for download: {e}"
                )

        else:

            st.warning(
                "The final image file could not be found."
            )

    else:

        st.warning(
            "No final image has been selected."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # START NEW SESSION
    # --------------------------------------------------------

    if st.button(
        "Start New Session",
        use_container_width=False,
    ):

        reset_session()

        st.rerun()


# ============================================================
# MAIN SESSION / ROUND VIEW
# ============================================================

else:

    # ========================================================
    # PLATEAU / NON-CONVERGENCE SCREEN
    # ========================================================

    if current_round > MAX_ROUNDS:

        st.title("Session Review")

        st.warning(
            f"""
            The maximum number of rounds ({MAX_ROUNDS}) has been
            reached. The system can now finalize the current face,
            restart the session, or fine-tune the face.
            """
        )

        st.markdown("### What would you like to do?")

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # FINALIZE
        # ----------------------------------------------------

        with col1:

            if st.button(
                "✅ Finalize Face",
                type="primary",
                use_container_width=True,
            ):

                final_path = (
                    st.session_state.get(
                        "latest_selected_image_path"
                    )
                    or st.session_state.get(
                        "selected_image_path"
                    )
                )

                st.session_state["final_image_path"] = final_path
                st.session_state["show_final_output"] = True

                st.rerun()

        # ----------------------------------------------------
        # RESTART
        # ----------------------------------------------------

        with col2:

            if st.button(
                "🔄 Restart Session",
                use_container_width=True,
            ):

                reset_session()

                st.rerun()

        # ----------------------------------------------------
        # FINE-TUNE
        # ----------------------------------------------------

        with col3:

            if st.button(
                "🎛️ Fine-tune",
                use_container_width=True,
            ):

                st.session_state["fine_tuning"] = True

                st.rerun()

        st.stop()

    # ========================================================
    # NORMAL ROUND VIEW
    # ========================================================

    st.title("RecallFace")

    st.subheader(
        f"Round {current_round}"
    )

    st.write(
        """
        Select the face that looks most similar to the person
        you are trying to identify.
        """
    )

    # ========================================================
    # HISTORY STRIP
    # ========================================================

    previous_rounds = []

    for key in st.session_state.keys():

        if key.startswith("round_"):

            try:
                round_number = int(
                    key.split("_")[1]
                )

                if round_number < current_round:
                    previous_rounds.append(
                        (
                            round_number,
                            st.session_state[key],
                        )
                    )

            except ValueError:
                pass

    previous_rounds.sort(
        key=lambda x: x[0]
    )

    if previous_rounds:

        st.markdown("### Session History")

        history_columns = st.columns(
            min(len(previous_rounds), 5)
        )

        for position, (
            round_number,
            round_data,
        ) in enumerate(previous_rounds):

            column = history_columns[
                position % len(history_columns)
            ]

            image_path = round_data.get(
                "selected_image_path"
            )

            selected_face = round_data.get(
                "selected_face"
            )

            with column:

                if (
                    image_path
                    and Path(image_path).exists()
                ):

                    st.image(
                        image_path,
                        caption=(
                            f"Round {round_number} "
                            f"• Face {selected_face}"
                        ),
                        width=140,
                    )

                else:

                    st.caption(
                        f"Round {round_number}"
                    )

        st.markdown("---")

    # ========================================================
    # ROUND INFORMATION
    # ========================================================

    if current_round == 1:

        st.info(
            "Round 1 uses the permanent Phase 2 starter set."
        )

    else:

        st.info(
            """
            This round is currently simulated using the Phase 2
            starter images in reverse order. Real generated candidates
            will be connected during Phase 11.
            """
        )

    # ========================================================
    # GET CANDIDATE IMAGES
    # ========================================================

    image_paths = get_round_images(
        current_round
    )

    if not image_paths:

        st.error(
            f"""
            No starter images were found.

            Expected directory:

            {STARTER_IMAGE_DIR}
            """
        )

        st.stop()

    # ========================================================
    # DISPLAY CANDIDATE GRID
    # ========================================================

    st.markdown(
        "### Select the closest-looking face"
    )

    columns = st.columns(4)

    for index, image_path in enumerate(
        image_paths
    ):

        column = columns[
            index % 4
        ]

        with column:

            st.image(
                str(image_path),
                width="stretch",
            )

            face_number = index + 1

            if st.button(
                f"Select Face {face_number}",
                key=(
                    f"select_"
                    f"{current_round}_"
                    f"{index}"
                ),
                use_container_width=True,
            ):

                st.session_state[
                    "selected_face"
                ] = face_number

                # Store exact selected image path.
                st.session_state[
                    "selected_image_path"
                ] = str(image_path)

                st.rerun()

    # ========================================================
    # SELECTION DETAILS
    # ========================================================

    if st.session_state["selected_face"]:

        st.markdown("---")

        st.success(
            f"""
            Face {st.session_state["selected_face"]}
            selected.
            """
        )

        selected_path = (
            st.session_state[
                "selected_image_path"
            ]
        )

        # ----------------------------------------------------
        # SHOW SELECTED IMAGE
        # ----------------------------------------------------

        if (
            selected_path
            and Path(selected_path).exists()
        ):

            st.markdown(
                "### Selected Face"
            )

            st.image(
                selected_path,
                width=350,
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = st.slider(
            "How confident are you in this selection?",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            format="%d%%",
        )

        # ----------------------------------------------------
        # OPTIONAL GUIDANCE
        # ----------------------------------------------------

        guidance = st.text_area(
            "Optional guidance",
            placeholder=(
                "Example: The eyes and face shape "
                "look similar..."
            ),
        )

        # ----------------------------------------------------
        # CONTINUE BUTTON
        # ----------------------------------------------------

        if st.button(
            "Continue to Next Round →",
            type="primary",
            use_container_width=False,
        ):

            # -----------------------------------------------
            # Store this round
            # -----------------------------------------------

            st.session_state[
                f"round_{current_round}"
            ] = {
                "selected_face": (
                    st.session_state[
                        "selected_face"
                    ]
                ),
                "selected_image_path": (
                    st.session_state[
                        "selected_image_path"
                    ]
                ),
                "confidence": confidence,
                "guidance": guidance,
            }

            # -----------------------------------------------
            # IMPORTANT FIX
            #
            # Preserve the latest selected image before
            # resetting the temporary selection state.
            # -----------------------------------------------

            st.session_state[
                "latest_selected_image_path"
            ] = (
                st.session_state[
                    "selected_image_path"
                ]
            )

            # -----------------------------------------------
            # Move to next round
            # -----------------------------------------------

            st.session_state[
                "current_round"
            ] += 1

            # -----------------------------------------------
            # Reset temporary selection
            # -----------------------------------------------

            st.session_state[
                "selected_face"
            ] = None

            st.session_state[
                "selected_image_path"
            ] = None

            st.rerun()