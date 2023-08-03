import os
os.environ["MPLBACKEND"] = "agg"
import streamlit as st
import PIL
from ultralytics import YOLO


st.set_page_config("Object detection",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )
with st.sidebar:
    st.header("Image/Video Config") #heading to the sidebar
    source_img = st.file_uploader("Choose an Image..",type = ("jpg","jpeg","png"))

st.title("Welcome to Object detection ARENA!")

col1,col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
model = YOLO("yolov8l.pt")

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:,:,::-1]
    with col2:
        st.image(res_plotted,caption="Detected Image",
                 use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
