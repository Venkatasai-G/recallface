from pathlib import Path

import sys



import streamlit as st





# ============================================================

# PROJECT ROOT

# ============================================================



PROJECT_ROOT = Path(__file__).resolve().parent.parent



if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))





# ============================================================

# PHASE 10 BACKEND

# ============================================================



from phase10_backend.database import (

    initialize_database,

    migrate_database,

)



from phase10_backend.session_manager import (

    create_session,

    save_round,

    get_session,

    get_session_rounds,

    finalize_session,

)







from phase11_integration.navigation_engine import NavigationEngine





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="RecallFace",

    page_icon="👤",

    layout="wide",

)





# ============================================================

# DATABASE INITIALIZATION

# ============================================================



initialize_database()

migrate_database()





# ============================================================

# PROJECT PATHS

# ============================================================



STARTER_IMAGE_DIR = (

    PROJECT_ROOT

    / "data"

    / "starter_set"

    / "images"

)



STARTER_LATENT_PATH = (

    PROJECT_ROOT

    / "data"

    / "starter_set"

    / "starter_latents.pt"

)



GENERATED_CANDIDATE_DIR = (

    PROJECT_ROOT

    / "data"

    / "generated_candidates"

)



MAX_ROUNDS = 5



# ============================================================

# PHASE 11 ENGINE AVAILABILITY

# ============================================================



try:

    import torch



    PHASE11_GPU_AVAILABLE = torch.cuda.is_available()

except Exception:

    PHASE11_GPU_AVAILABLE = False



# ============================================================

# SESSION STATE INITIALIZATION

# ============================================================



if "session_started" not in st.session_state:

    st.session_state["session_started"] = False



if "session_id" not in st.session_state:

    st.session_state["session_id"] = None



if "selected_face" not in st.session_state:

    st.session_state["selected_face"] = None



if "selected_image_path" not in st.session_state:

    st.session_state["selected_image_path"] = None



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



if "current_latent" not in st.session_state:

    st.session_state["current_latent"] = None



if "candidate_latents" not in st.session_state:

    st.session_state["candidate_latents"] = None



if "candidate_images" not in st.session_state:

    st.session_state["candidate_images"] = None



if "candidate_image_paths" not in st.session_state:

    st.session_state["candidate_image_paths"] = []



if "navigation_engine" not in st.session_state:

    st.session_state["navigation_engine"] = None



current_round = st.session_state["current_round"]



# ============================================================

# PHASE 11 NAVIGATION ENGINE

# ============================================================



if (

    PHASE11_GPU_AVAILABLE

    and st.session_state["navigation_engine"] is None

):

    try:

        st.session_state["navigation_engine"] = NavigationEngine(

            device="cuda",

            num_candidates=12,

            latent_clip=3.0,

        )

    except Exception as e:

        st.session_state["navigation_engine"] = None

        st.warning(

            f"Phase 11 engine could not be loaded: {e}"

        )



# ============================================================

# HELPER FUNCTIONS

# ============================================================



def reset_session():

    """

    Reset the current Streamlit session state.

    """



    keys_to_reset = [

        "session_started",

        "session_id",

        "selected_face",

        "selected_image_path",

        "latest_selected_image_path",

        "current_round",

        "fine_tuning",

        "show_final_output",

        "final_image_path",

        "fine_tuning_values",

        "fine_tuning_complete",

        "current_latent",

        "candidate_latents",

        "candidate_images",

        "candidate_image_paths",

    ]



    for key in keys_to_reset:



        if key == "session_started":

            st.session_state[key] = False



        elif key == "current_round":

            st.session_state[key] = 1



        elif key in [

            "session_id",

            "selected_face",

            "selected_image_path",

            "latest_selected_image_path",

            "final_image_path",

        ]:

            st.session_state[key] = None



        elif key == "fine_tuning_values":

            st.session_state[key] = {}



        elif key == "candidate_image_paths":

            st.session_state[key] = []



        elif key in [

            "current_latent",

            "candidate_latents",

            "candidate_images",

        ]:

            st.session_state[key] = None



        else:

            st.session_state[key] = False



def get_starter_images():

    """

    Return Phase 2 starter images.

    """



    if not STARTER_IMAGE_DIR.exists():

        return []



    image_paths = sorted(

        list(STARTER_IMAGE_DIR.glob("*.png"))

        + list(STARTER_IMAGE_DIR.glob("*.jpg"))

        + list(STARTER_IMAGE_DIR.glob("*.jpeg"))

    )



    return image_paths



def get_starter_latents():

    """

    Load the 12 Phase 2 starter latent vectors.

    """



    if not STARTER_LATENT_PATH.exists():

        return None



    import torch



    latents = torch.load(

        STARTER_LATENT_PATH,

        map_location="cpu",

        weights_only=False,

    )



    if not isinstance(latents, torch.Tensor):

        return None



    if latents.shape != (12, 512):

        return None



    return latents



def get_starter_latent(face_number):

    """

    Return the Phase 2 latent corresponding to a starter face number.

    """



    latents = get_starter_latents()



    if latents is None:

        return None



    if not 1 <= face_number <= len(latents):

        return None



    return latents[face_number - 1].clone()



def generate_phase11_candidates():

    """

    Generate candidates using the Phase 11 NavigationEngine.



    Returns:

        tuple:

            candidate_latents

            candidate_images

    """



    engine = st.session_state.get("navigation_engine")

    current_latent = st.session_state.get("current_latent")

    round_number = st.session_state.get("current_round")



    if engine is None:

        return None, None



    if current_latent is None:

        return None, None



    if round_number is None or round_number <= 1:

        return None, None



    candidate_latents, candidate_images = engine.generate_round(

        current_latent=current_latent,

        round_number=round_number,

    )



    return candidate_latents, candidate_images



def save_candidate_images(candidate_images, round_number):

    """

    Save Phase 11 generated candidate images as PNG files.



    Returns:

        list[Path]: Paths to the saved candidate images.

    """



    if candidate_images is None:

        return []



    session_id = st.session_state.get("session_id")



    if not session_id:

        return []



    round_dir = (

        GENERATED_CANDIDATE_DIR

        / str(session_id)

        / f"round_{round_number}"

    )



    round_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    image_paths = []



    from PIL import Image

    import numpy as np



    for index, image_array in enumerate(candidate_images):

        image_array = np.asarray(

            image_array,

            dtype=np.uint8,

        )



        image = Image.fromarray(

            image_array,

            mode="RGB",

        )



        image_path = (

            round_dir

            / f"candidate_{index + 1}.png"

        )



        image.save(image_path)



        image_paths.append(image_path)



    return image_paths



def get_round_images(round_number):

    """

    Return candidate images for the current round.



    Round 1:

        Use the permanent Phase 2 starter set.



    Later rounds:

        Use the real Phase 11 NavigationEngine when available.

        On CPU-only environments, fall back to the Phase 2

        starter images so the Streamlit UI remains usable.

    """



    starter_images = get_starter_images()



    if round_number == 1:

        return starter_images



     # Reuse candidates already generated for this round.

    existing_paths = st.session_state.get(

        "candidate_image_paths"

    )



    if existing_paths:

        existing_paths = [

            Path(path)

            for path in existing_paths

            if Path(path).exists()

        ]



        if existing_paths:

            return existing_paths



    # --------------------------------------------------------

    # Phase 11 real candidate generation

    # --------------------------------------------------------



    if PHASE11_GPU_AVAILABLE:

        try:

            candidate_latents, candidate_images = (

                generate_phase11_candidates()

            )



            if (

                candidate_latents is not None

                and candidate_images is not None

            ):

                st.session_state[

                    "candidate_latents"

                ] = candidate_latents



                st.session_state[

                    "candidate_images"

                ] = candidate_images



                candidate_image_paths = save_candidate_images(

                    candidate_images,

                    round_number,

                )



                st.session_state[

                    "candidate_image_paths"

                ] = candidate_image_paths



                return candidate_image_paths



        except Exception as e:

            st.warning(

                f"Phase 11 candidate generation failed: {e}"

            )



    # --------------------------------------------------------

    # CPU/local development fallback

    # --------------------------------------------------------



    return list(reversed(starter_images))



def get_saved_rounds():

    """

    Retrieve the current session's rounds from SQLite.

    """



    session_id = st.session_state.get("session_id")



    if not session_id:

        return []



    return get_session_rounds(session_id)



def finalize_current_session():

    """

    Mark the current database session as completed.

    """



    session_id = st.session_state.get("session_id")



    if session_id:

        finalize_session(session_id)



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

        """

        Round 1 uses the permanent Phase 2 starter set.

        On GPU-enabled environments, later rounds use the trained

        Phase 11 Projector/Sampler with DiffAE candidate generation.

        CPU-only environments use the local development fallback.

        """

    )



    if st.button(

        "🚀 Start Session",

        type="primary",

        use_container_width=False,

    ):



        # ----------------------------------------------------

        # CREATE DATABASE SESSION

        # ----------------------------------------------------



        session_id = create_session()



        st.session_state["session_id"] = session_id

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

        st.session_state["current_latent"] = None

        st.session_state["candidate_latents"] = None

        st.session_state["candidate_images"] = None

        st.session_state["candidate_image_paths"] = []



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

        )



        hair = st.slider(

            "Hair",

            min_value=-5,

            max_value=5,

            value=0,

            step=1,

        )



        face_shape = st.slider(

            "Face Shape",

            min_value=-5,

            max_value=5,

            value=0,

            step=1,

        )



    with col2:



        smile = st.slider(

            "Smile",

            min_value=-5,

            max_value=5,

            value=0,

            step=1,

        )



        eyes = st.slider(

            "Eyes",

            min_value=-5,

            max_value=5,

            value=0,

            step=1,

        )



    st.markdown("---")



    # ========================================================

    # CURRENT COMPOSITE

    # ========================================================



    current_image = (

        st.session_state.get(

            "latest_selected_image_path"

        )

        or st.session_state.get(

            "selected_image_path"

        )

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



    # ========================================================

    # BUTTONS

    # ========================================================



    col1, col2, col3 = st.columns(3)



    with col1:



        if st.button(

            "Apply Fine-tuning",

            type="primary",

            use_container_width=True,

        ):



            st.session_state[

                "fine_tuning_values"

            ] = {

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



            final_path = (

                st.session_state.get(

                    "latest_selected_image_path"

                )

                or st.session_state.get(

                    "selected_image_path"

                )

            )



            st.session_state[

                "final_image_path"

            ] = final_path



            st.session_state[

                "fine_tuning_complete"

            ] = True



            # ------------------------------------------------

            # COMPLETE DATABASE SESSION

            # ------------------------------------------------



            finalize_current_session()



            st.session_state[

                "fine_tuning"

            ] = False



            st.session_state[

                "show_final_output"

            ] = True



            st.rerun()



    with col3:



        if st.button(

            "← Back to Session Review",

            use_container_width=True,

        ):



            st.session_state[

                "fine_tuning"

            ] = False



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



    # ========================================================

    # FINAL IMAGE

    # ========================================================



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

                    f"Unable to prepare image for download: {e}"

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



    # ========================================================

    # START NEW SESSION

    # ========================================================



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

            The maximum number of rounds ({MAX_ROUNDS}) has

            been reached. The system can now finalize the

            current face, restart the session, or fine-tune

            the face.

            """

        )



        # ====================================================

        # DATABASE SESSION INFORMATION

        # ====================================================



        session_id = st.session_state.get(

            "session_id"

        )



        if session_id:



            session_data = get_session(

                session_id

            )



            if session_data:



                st.caption(

                    f"Session ID: {session_data['session_id']}"

                )



                st.caption(

                    f"Status: {session_data['status']}"

                )



        st.markdown(

            "### What would you like to do?"

        )



        col1, col2, col3 = st.columns(3)



        # ====================================================

        # FINALIZE

        # ====================================================



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



                st.session_state[

                    "final_image_path"

                ] = final_path



                # Complete database session

                finalize_current_session()



                st.session_state[

                    "show_final_output"

                ] = True



                st.rerun()



        # ====================================================

        # RESTART

        # ====================================================



        with col2:



            if st.button(

                "🔄 Restart Session",

                use_container_width=True,

            ):



                reset_session()



                st.rerun()



        # ====================================================

        # FINE-TUNE

        # ====================================================



        with col3:



            if st.button(

                "🎛️ Fine-tune",

                use_container_width=True,

            ):



                st.session_state[

                    "fine_tuning"

                ] = True



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

    # SESSION INFORMATION

    # ========================================================



    session_id = st.session_state.get(

        "session_id"

    )



    if session_id:



        st.caption(

            f"Session ID: {session_id}"

        )



    # ========================================================

    # HISTORY STRIP

    # ========================================================



    saved_rounds = get_saved_rounds()



    if saved_rounds:



        st.markdown(

            "### Session History"

        )



        history_columns = st.columns(

            min(len(saved_rounds), 5)

        )



        for position, round_data in enumerate(

            saved_rounds

        ):



            column = history_columns[

                position % len(history_columns)

            ]



            image_path = round_data.get(

                "selected_image_path"

            )



            selected_face = round_data.get(

                "selected_face"

            )



            round_number = round_data.get(

                "round_number"

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

            This round uses the Phase 11 NavigationEngine when a

            CUDA-enabled environment is available. On CPU-only

            environments, the Phase 2 starter images are used as a

            local development fallback.

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



                st.session_state["selected_image_path"] = str(image_path)

                # Store the latent corresponding to the selected candidate.
                if current_round == 1:
                    # Round 1 uses the permanent Phase 2 starter set.
                    selected_latent = get_starter_latent(face_number)

                    if selected_latent is not None:
                        st.session_state["current_latent"] = selected_latent

                else:
                    # Later rounds use candidates generated by the Phase 11
                    # NavigationEngine. The selected candidate becomes the
                    # starting latent for the next round.
                    candidate_latents = st.session_state.get("candidate_latents")

                    if candidate_latents is not None:
                        selected_index = face_number - 1

                        if 0 <= selected_index < len(candidate_latents):
                            selected_latent = candidate_latents[selected_index]
                            st.session_state["current_latent"] = (
                                selected_latent.detach().cpu().clone()
                            )

                st.rerun()



    # ========================================================

    # SELECTION DETAILS

    # ========================================================



    if st.session_state[

        "selected_face"

    ]:



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



        # ====================================================

        # SELECTED IMAGE

        # ====================================================



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



        # ====================================================

        # CONFIDENCE

        # ====================================================



        confidence = st.slider(

            "How confident are you in this selection?",

            min_value=0,

            max_value=100,

            value=50,

            step=5,

            format="%d%%",

        )



        # ====================================================

        # GUIDANCE

        # ====================================================



        guidance = st.text_area(

            "Optional guidance",

            placeholder=(

                "Example: The eyes and face shape "

                "look similar..."

            ),

        )



        # ====================================================

        # CONTINUE

        # ====================================================



        if st.button(

            "Continue to Next Round →",

            type="primary",

            use_container_width=False,

        ):



            # -----------------------------------------------

            # Get current values BEFORE clearing state

            # -----------------------------------------------



            selected_face = (

                st.session_state[

                    "selected_face"

                ]

            )



            selected_image_path = (

                st.session_state[

                    "selected_image_path"

                ]

            )



            round_number = current_round



            # -----------------------------------------------

            # SAVE ROUND TO SQLITE

            # -----------------------------------------------



            save_round(

                session_id=st.session_state[

                    "session_id"

                ],

                round_number=round_number,

                selected_face=selected_face,

                selected_image_path=selected_image_path,

                confidence=confidence,

                guidance=guidance,

            )



            # -----------------------------------------------

            # Preserve latest selected image

            # -----------------------------------------------



            st.session_state[

                "latest_selected_image_path"

            ] = selected_image_path



            # -----------------------------------------------

            # Also keep temporary round information

            # for compatibility with the Phase 9 UI.

            # -----------------------------------------------



            st.session_state[

                f"round_{round_number}"

            ] = {

                "selected_face": selected_face,

                "selected_image_path": selected_image_path,

                "confidence": confidence,

                "guidance": guidance,

            }



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



            st.session_state[

                "candidate_latents"

            ] = None



            st.session_state[

                "candidate_images"

            ] = None



            st.session_state[

                "candidate_image_paths"

            ] = []



            st.rerun()