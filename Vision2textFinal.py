# """
# Universal Image-to-Text Analyzer (Post-Analysis Translation)
# -----------------------------------------------------------
# 1) Gemini does the entire image-based analysis in English:
#    - OCR + translation to English
#    - Visual description
#    - Answers a user question
#    - Provides references
# 2) Then optionally translate that entire Gemini result into
#    one of many languages (including Greek) using GPT-4o.

# Author: <your-name>
# """

# import os
# from pathlib import Path

# import streamlit as st
# from PIL import Image as PILImage

# # Optional local OCR with Tesseract
# try:
#     import pytesseract
#     _OCR_AVAILABLE = True
# except ImportError:
#     _OCR_AVAILABLE = False

# # agno for Gemini
# from agno.agent import Agent
# from agno.models.google import Gemini
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.media import Image as AgnoImage

# # OpenAI ≥ 1.0 client for GPT-4/4o translation
# try:
#     from openai import OpenAI  # new-style client
#     _OPENAI_AVAILABLE = True
# except ImportError:
#     _OPENAI_AVAILABLE = False


# # ────────────────────────────────────────────────────────────────────────────────
# # Streamlit page config
# # ────────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Universal Image-to-Text Analyzer with GPT-4o Translation",
#     page_icon="🖼️",
#     layout="centered"
# )

# if "GOOGLE_API_KEY" not in st.session_state:
#     st.session_state.GOOGLE_API_KEY = None
# if "OPENAI_API_KEY" not in st.session_state:
#     st.session_state.OPENAI_API_KEY = None

# # ────────────────────────────────────────────────────────────────────────────────
# # Sidebar config - keys, help
# # ────────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.title("🔑 Configuration")

#     # Google key for Gemini
#     if not st.session_state.GOOGLE_API_KEY:
#         g_key = st.text_input(
#             "Google AI Key (Gemini)",
#             type="password",
#             help="From: https://aistudio.google.com/apikey"
#         )
#         if g_key:
#             st.session_state.GOOGLE_API_KEY = g_key.strip()
#             st.success("Gemini key saved. Reloading ...")
#             st.rerun()
#     else:
#         st.success("✅ Gemini key configured")
#         if st.button("Reset Google Key"):
#             st.session_state.GOOGLE_API_KEY = None
#             st.rerun()

#     # OpenAI key for GPT-4o
#     if not st.session_state.OPENAI_API_KEY:
#         oa_key = st.text_input(
#             "OpenAI Key (GPT-4o)",
#             type="password",
#             help="Create at https://platform.openai.com/; needed for translations."
#         )
#         if oa_key:
#             st.session_state.OPENAI_API_KEY = oa_key.strip()
#             st.success("OpenAI key saved. Reloading ...")
#             st.rerun()
#     else:
#         st.success("✅ OpenAI key configured")
#         if st.button("Reset OpenAI Key"):
#             st.session_state.OPENAI_API_KEY = None
#             st.rerun()

#     st.info(
#         "1) Upload image, Gemini does analysis in English.\n"
#         "2) (Optional) Translate the entire result to another language with GPT-4o.\n"
#     )
#     st.warning(
#         "⚠️ AI output can contain errors. Remove sensitive data first."
#     )

# # ────────────────────────────────────────────────────────────────────────────────
# # Halt if no Gemini key
# # ────────────────────────────────────────────────────────────────────────────────
# if not st.session_state.GOOGLE_API_KEY:
#     st.stop()

# # Configure OpenAI client if present
# client = None
# if _OPENAI_AVAILABLE and st.session_state.OPENAI_API_KEY:
#     client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

# # ────────────────────────────────────────────────────────────────────────────────
# # Build Gemini agent
# # ────────────────────────────────────────────────────────────────────────────────
# image_agent = Agent(
#     model=Gemini(
#         id="gemini-2.0-flash",
#         api_key=st.session_state.GOOGLE_API_KEY
#     ),
#     tools=[DuckDuckGoTools()],
#     markdown=True
# )

# # ────────────────────────────────────────────────────────────────────────────────
# # Page layout containers
# # ────────────────────────────────────────────────────────────────────────────────
# st.title("🖼️ Universal Image-to-Text Analyzer")
# st.caption("Gemini analysis in English → optional GPT-4o translations to multiple languages")

# upload_container = st.container()
# prompt_container = st.container()
# display_container = st.container()
# analysis_container = st.container()
# translation_container = st.container()

# # ────────────────────────────────────────────────────────────────────────────────
# # Upload widget
# # ────────────────────────────────────────────────────────────────────────────────
# with upload_container:
#     uploaded_file = st.file_uploader(
#         "📤 Upload an image (JPG, PNG, GIF, TIFF, BMP, WEBP, DICOM)",
#         type=["jpg", "jpeg", "png", "gif", "tiff", "bmp", "webp", "dicom"]
#     )

# # ────────────────────────────────────────────────────────────────────────────────
# # User question
# # ────────────────────────────────────────────────────────────────────────────────
# with prompt_container:
#     user_question = st.text_area(
#         "✍️ Ask a question about this image (optional)",
#         placeholder="E.g. 'What brand of cereal is on the shelf?' or 'Summarize the labels.'",
#         height=100
#     )

# # ────────────────────────────────────────────────────────────────────────────────
# # Show image & analyze button
# # ────────────────────────────────────────────────────────────────────────────────
# if uploaded_file:
#     with display_container:
#         image = PILImage.open(uploaded_file)
#         max_w = 512
#         w, h = image.size
#         if w > max_w:
#             h = round(max_w / w * h)
#             w = max_w
#             image = image.resize((w, h))

#         st.image(image, caption="📷 Uploaded image preview", use_container_width=False)
#         analyze_clicked = st.button("🔍 Analyze Image")
# else:
#     st.info("👆 Upload an image to enable analysis.")
#     analyze_clicked = False

# # ────────────────────────────────────────────────────────────────────────────────
# # Local OCR function
# # ────────────────────────────────────────────────────────────────────────────────
# def run_tesseract_ocr(pil_img: PILImage.Image) -> str:
#     if not _OCR_AVAILABLE:
#         return ""
#     return pytesseract.image_to_string(pil_img.convert("RGB"), lang="eng")

# # ────────────────────────────────────────────────────────────────────────────────
# # Gemini system prompt
# # ────────────────────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT = """\
# You are an advanced multimodal assistant that:
# 1. **Transcribes** all visible text from the image.
# 2. **Translates** that text to English.
# 3. **Describes** the visual scene in well-structured Markdown.
# 4. **Answers** the user's question if any.
# 5. **Cites** 2-3 references via DuckDuckGo.

# Format:
# ### 1. OCR Transcription
# (Detected raw text)

# ### 2. OCR Translation (→ English)
# (English version of that text)

# ### 3. Visual Description
# (Objects, context, etc.)

# ### 4. Answer to User's Prompt
# (If no question, say so.)

# ### 5. References
# [List 2-3 references as bullet points with inline links]
# """

# analysis_result = None

# if analyze_clicked and uploaded_file:
#     with analysis_container:
#         with st.spinner("🧠 Gemini analyzing ..."):
#             tmp_path = Path("temp_upload.png")
#             image.save(tmp_path)

#             local_ocr = run_tesseract_ocr(PILImage.open(tmp_path))
#             context_block = f"---\nLocal OCR:\n{local_ocr}\n---" if local_ocr else ""

#             user_part = user_question.strip() if user_question else ""
#             final_query = "\n\n".join(filter(None, [SYSTEM_PROMPT, context_block, user_part]))

#             try:
#                 response = image_agent.run(final_query, images=[AgnoImage(filepath=str(tmp_path))])
#                 analysis_result = response.content
#             except Exception as ex:
#                 st.error(f"Gemini error: {ex}")
#                 analysis_result = None

#         if analysis_result:
#             st.markdown("## 📑 Analysis Result (English)")
#             st.markdown(analysis_result)
#             st.session_state["analysis_result"] = analysis_result
#         else:
#             st.info("No result from Gemini.")

# # ────────────────────────────────────────────────────────────────────────────────
# # Fallback notice if no Tesseract
# # ────────────────────────────────────────────────────────────────────────────────
# if not _OCR_AVAILABLE:
#     st.info("ℹ️ Tesseract is not installed. We'll rely on Gemini's OCR. For better accuracy, install Tesseract + `pytesseract`.")

# # ────────────────────────────────────────────────────────────────────────────────
# # Post-Analysis Translation
# # ────────────────────────────────────────────────────────────────────────────────
# with translation_container:
#     st.markdown("---")
#     st.subheader("🌐 Translate the Gemini Result?")

#     if "analysis_result" not in st.session_state or not st.session_state["analysis_result"]:
#         st.info("Analyze an image first to enable translation.")
#     else:
#         if client is None:
#             st.info("Provide an OpenAI key for GPT-4/4o to enable translation.")
#         else:
#             languages = [
#                 "(Select Language)",
#                 "Spanish", "French", "German", "Chinese", "Arabic",
#                 "Hindi", "Portuguese", "Greek", "Japanese", "Korean",
#                 "Russian", "Italian"
#             ]
#             lang_choice = st.selectbox("Choose a language:", languages, index=0)
#             do_translate = st.button("Translate to Selected Language")

#             if lang_choice != "(Select Language)" and do_translate:
#                 with st.spinner(f"Translating to {lang_choice}..."):
#                     try:
#                         original_text = st.session_state["analysis_result"]

#                         translation_resp = client.chat.completions.create(
#                             model="gpt-4o-mini",  # change to "gpt-4o" if you have full access
#                             messages=[
#                                 {
#                                     "role": "system",
#                                     "content": (
#                                         f"You are a multilingual translator. "
#                                         f"Translate the following text from English into {lang_choice}, "
#                                         "preserving Markdown formatting. "
#                                         "Respond with ONLY the translated text."
#                                     )
#                                 },
#                                 {"role": "user", "content": original_text},
#                             ],
#                             temperature=0.0
#                         )
#                         translated_output = translation_resp.choices[0].message.content.strip()
#                         st.markdown("## 🌐 Translated Output")
#                         st.markdown(translated_output)
#                     except Exception as ex:
#                         st.error(f"Translation error: {ex}")


"""
Universal Image-to-Text Analyzer (Post-Analysis Translation)
-----------------------------------------------------------
1) Gemini does the entire image-based analysis in English:
   - OCR + translation to English
   - Visual description
   - Answers a user question
   - Provides references
2) Then optionally translate that entire Gemini result into
   one of many languages (including Greek) using GPT-4.

Author: <your-name>
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image as PILImage

# Optional local OCR with Tesseract
try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# agno for Gemini
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# openai for GPT-4 translation
try:
    from openai import OpenAI  # new-style client in openai>=1.0
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

###################################
#  PDF support (added code block) #
###################################
try:
    from pdf2image import convert_from_bytes
    _PDF_SUPPORT = True
except ImportError:
    _PDF_SUPPORT = False
    # We skip PDF decomposition if pdf2image not installed

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Image-to-Text Analyzer with GPT-4o Translation",
    page_icon="🖼️",
    layout="centered"
)

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar config - keys, help
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔑 Configuration")

    # Google key for Gemini
    if not st.session_state.GOOGLE_API_KEY:
        g_key = st.text_input(
            "Google AI Key (Gemini)",
            type="password",
            help="From: https://aistudio.google.com/apikey"
        )
        if g_key:
            st.session_state.GOOGLE_API_KEY = g_key.strip()
            st.success("Gemini key saved. Reloading ...")
            st.rerun()
    else:
        st.success("✅ Gemini key configured")
        if st.button("Reset Google Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

    # OpenAI key for GPT-4o
    if not st.session_state.OPENAI_API_KEY:
        oa_key = st.text_input(
            "OpenAI Key (GPT-4o)",
            type="password",
            help="Create at https://platform.openai.com/; needed for translations."
        )
        if oa_key:
            st.session_state.OPENAI_API_KEY = oa_key.strip()
            st.success("OpenAI key saved. Reloading ...")
            st.rerun()
    else:
        st.success("✅ OpenAI key configured")
        if st.button("Reset OpenAI Key"):
            st.session_state.OPENAI_API_KEY = None
            st.rerun()

    st.info(
        "1) Upload image or PDF, Gemini does analysis in English.\n"
        "2) (Optional) Translate the entire result to another language with GPT-4o.\n"
    )
    st.warning(
        "⚠️ AI output can contain errors. Remove sensitive data first."
    )

# ────────────────────────────────────────────────────────────────────────────────
# Halt if no Gemini key
# ────────────────────────────────────────────────────────────────────────────────
if not st.session_state.GOOGLE_API_KEY:
    st.stop()

# Configure OpenAI client if present
client = None
if _OPENAI_AVAILABLE and st.session_state.OPENAI_API_KEY:
    client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# Build Gemini agent
# ────────────────────────────────────────────────────────────────────────────────
image_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# ────────────────────────────────────────────────────────────────────────────────
# Page layout containers
# ────────────────────────────────────────────────────────────────────────────────
st.title("🖼️ Universal Image-to-Text Analyzer")
st.caption("Gemini analysis in English → optional GPT-4o translations to multiple languages")

upload_container = st.container()
prompt_container = st.container()
display_container = st.container()
analysis_container = st.container()
translation_container = st.container()

# ────────────────────────────────────────────────────────────────────────────────
# Upload widget
# ────────────────────────────────────────────────────────────────────────────────
with upload_container:
    uploaded_file = st.file_uploader(
        "📤 Upload an image/PDF (JPG, PNG, GIF, TIFF, BMP, WEBP, DICOM, PDF)",
        type=["jpg", "jpeg", "png", "gif", "tiff", "bmp", "webp", "dicom", "pdf"]
    )

# ────────────────────────────────────────────────────────────────────────────────
# User question
# ────────────────────────────────────────────────────────────────────────────────
with prompt_container:
    user_question = st.text_area(
        "✍️ Ask a question about this document (optional)",
        placeholder="E.g. 'What brand of cereal is on the shelf?' or 'Summarize the text in the PDF.'",
        height=100
    )

# ────────────────────────────────────────────────────────────────────────────────
# Show & analyze button
# ────────────────────────────────────────────────────────────────────────────────
analyze_clicked = False
if uploaded_file:
    file_type = uploaded_file.type.lower()
    if "pdf" in file_type:
        st.info("Uploaded PDF detected. Pages will be analyzed individually.")
        analyze_clicked = st.button("🔍 Analyze PDF")
    else:
        with display_container:
            image = PILImage.open(uploaded_file)
            max_w = 512
            w, h = image.size
            if w > max_w:
                h = round(max_w / w * h)
                w = max_w
                image = image.resize((w, h))

            st.image(image, caption="📷 Uploaded image preview", use_container_width=False)
            analyze_clicked = st.button("🔍 Analyze Image")
else:
    st.info("👆 Upload an image/PDF to enable analysis.")

# ────────────────────────────────────────────────────────────────────────────────
# Local OCR function
# ────────────────────────────────────────────────────────────────────────────────
def run_tesseract_ocr(pil_img: PILImage.Image) -> str:
    if not _OCR_AVAILABLE:
        return ""
    return pytesseract.image_to_string(pil_img.convert("RGB"), lang="eng")

# ────────────────────────────────────────────────────────────────────────────────
# Gemini system prompt
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an advanced multimodal assistant that:
1. **Transcribes** all visible text from the page/image.
2. **Translates** that text to English.
3. **Describes** the visual/structural content in well-structured Markdown.
4. **Answers** the user's question if any.
5. **Cites** 2-3 references via DuckDuckGo.

Format:
### 1. OCR Transcription
(Detected raw text)

### 2. OCR Translation (→ English)
(English version of that text)

### 3. Visual Description
(Objects, context, layout, etc.)

### 4. Answer to User's Prompt
(If no question, say so.)

### 5. References
[List 2-3 references as bullet points with inline links]
"""

analysis_result = None

# ────────────────────────────────────────────────────────────────────────────────
# PDF Decomposition (Added Code)
# ────────────────────────────────────────────────────────────────────────────────
def decompose_pdf_into_pages(uploaded_bytes) -> list[PILImage.Image]:
    """
    Returns a list of PIL Images, one per PDF page, if pdf2image is installed.
    Otherwise returns an empty list.
    """
    if not _PDF_SUPPORT:
        return []
    return convert_from_bytes(uploaded_bytes, dpi=150)


# ────────────────────────────────────────────────────────────────────────────────
# Analysis step
# ────────────────────────────────────────────────────────────────────────────────
def analyze_image_page(pil_img: PILImage.Image, question: str) -> str:
    """
    Takes a PIL image, optional user question, and runs the same Gemini logic
    as the original single-image approach. Returns the analysis result (string).
    """
    # Save image to temp
    tmp_path = Path("temp_upload_page.png")
    pil_img.save(tmp_path)

    # Local OCR
    local_ocr = run_tesseract_ocr(pil_img)
    context_block = f"---\nLocal OCR:\n{local_ocr}\n---" if local_ocr else ""

    user_part = question.strip() if question else ""
    final_query = "\n\n".join(filter(None, [SYSTEM_PROMPT, context_block, user_part]))

    try:
        response = image_agent.run(final_query, images=[AgnoImage(filepath=str(tmp_path))])
        return response.content
    except Exception as ex:
        st.error(f"Gemini error: {ex}")
        return ""


if analyze_clicked and uploaded_file:
    with analysis_container:
        with st.spinner("🧠 Analyzing ..."):
            file_type = uploaded_file.type.lower()
            # If PDF, break into pages and analyze each. If image, just do 1 pass.
            if "pdf" in file_type:
                if _PDF_SUPPORT:
                    pdf_bytes = uploaded_file.read()
                    pages = decompose_pdf_into_pages(pdf_bytes)
                    if not pages:
                        st.error("Could not decode PDF pages (missing pdf2image/Poppler?).")
                    else:
                        all_page_results = []
                        for i, page_img in enumerate(pages, start=1):
                            st.markdown(f"### Page {i} Analysis")
                            result = analyze_image_page(page_img, user_question)
                            st.markdown(result)
                            all_page_results.append(f"**Page {i}**:\n{result}\n\n")
                        # Store combined results
                        analysis_result = "\n".join(all_page_results)
                        st.session_state["analysis_result"] = analysis_result
                else:
                    st.error("PDF support not available (pdf2image not installed).")
            else:
                # Single image
                image = PILImage.open(uploaded_file)
                single_result = analyze_image_page(image, user_question)
                if single_result:
                    st.markdown("## 📑 Analysis Result (English)")
                    st.markdown(single_result)
                    analysis_result = single_result
                    st.session_state["analysis_result"] = analysis_result
                else:
                    st.info("No result from Gemini.")


# ────────────────────────────────────────────────────────────────────────────────
# Fallback notice if no Tesseract
# ────────────────────────────────────────────────────────────────────────────────
if not _OCR_AVAILABLE:
    st.info("ℹ️ Tesseract is not installed. We'll rely on Gemini's OCR. For better accuracy, install Tesseract + `pytesseract`.")


# ────────────────────────────────────────────────────────────────────────────────
# Post-Analysis Translation
# ────────────────────────────────────────────────────────────────────────────────
with translation_container:
    st.markdown("---")
    st.subheader("🌐 Translate the Gemini Result?")

    if "analysis_result" not in st.session_state or not st.session_state["analysis_result"]:
        st.info("Analyze an image or PDF first to enable translation.")
    else:
        if client is None:
            st.info("Provide an OpenAI key for GPT-4/4o to enable translation.")
        else:
            languages = [
                "(Select Language)",
                "Spanish", "French", "German", "Chinese", "Arabic",
                "Hindi", "Portuguese", "Greek", "Japanese", "Korean",
                "Russian", "Italian"
            ]
            lang_choice = st.selectbox("Choose a language:", languages, index=0)
            do_translate = st.button("Translate to Selected Language")

            if lang_choice != "(Select Language)" and do_translate:
                with st.spinner(f"Translating to {lang_choice}..."):
                    try:
                        original_text = st.session_state["analysis_result"]
                        translation_resp = client.chat.completions.create(
                            model="gpt-4o-mini",  # or "gpt-4o" if you have full model
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        f"You are a multilingual translator. "
                                        f"Translate the following text from English into {lang_choice}, "
                                        "preserving Markdown formatting. "
                                        "Respond with ONLY the translated text."
                                    )
                                },
                                {"role": "user", "content": original_text},
                            ],
                            temperature=0.0
                        )
                        translated_output = translation_resp.choices[0].message.content.strip()
                        st.markdown("## 🌐 Translated Output")
                        st.markdown(translated_output)
                    except Exception as ex:
                        st.error(f"Translation error: {ex}")
