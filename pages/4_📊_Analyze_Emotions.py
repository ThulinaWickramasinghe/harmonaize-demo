import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import cv2
from deepface import DeepFace
import tempfile
import os

st.set_page_config(page_title="Facial Expression Analysis", page_icon="📊")

st.markdown("# Facial Expression Analysis")
st.sidebar.header("Facial Expression Analysis")
st.write(
    """This demo shows how to analyze user facila expresions while singing."""
)

with tempfile.TemporaryDirectory() as temp_dir:
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video temporarily
        video_path = os.path.join(temp_dir, "temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Open the video using OpenCV
        video = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Create video writer for output
        output_path = os.path.join(temp_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process the video frames
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Perform facial expression analysis on the frame
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # Get the dominant emotion
                dominant_emotion = result[0]['dominant_emotion']

                # Draw the emotion text on the frame
                cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                # If no face is detected, skip the frame
                pass

            # Write the frame to the output video
            output_video.write(frame)

        # Release the video capture and writer objects
        video.release()
        output_video.release()

        # Display the output video
        st.video(output_path)

# @st.cache_data
# def get_UN_data():
#     AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
#     df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
#     return df.set_index("Region")


# try:
#     df = get_UN_data()
#     countries = st.multiselect(
#         "Choose countries", list(df.index), ["China", "United States of America"]
#     )
#     if not countries:
#         st.error("Please select at least one country.")
#     else:
#         data = df.loc[countries]
#         data /= 1000000.0
#         st.write("### Gross Agricultural Production ($B)", data.sort_index())

#         data = data.T.reset_index()
#         data = pd.melt(data, id_vars=["index"]).rename(
#             columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
#         )
#         chart = (
#             alt.Chart(data)
#             .mark_area(opacity=0.3)
#             .encode(
#                 x="year:T",
#                 y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
#                 color="Region:N",
#             )
#         )
#         st.altair_chart(chart, use_container_width=True)
# except URLError as e:
#     st.error(
#         """
#         **This demo requires internet access.**
#         Connection error: %s
#     """
#         % e.reason
#     )
