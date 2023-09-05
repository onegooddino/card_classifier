import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Card classifier")

st.write("Predict the playing card that is being represented in the image.")

model = load_model("card_v1.h5")
labels = ['ace of clubs','ace of diamonds','ace of hearts','ace of spades','eight of clubs','eight of diamonds','eight of hearts','eight of spades','five of clubs','five of diamonds','five of hearts','five of spades','four of clubs','four of diamonds','four of hearts','four of spades','jack of clubs','jack of diamonds','jack of hearts','jack of spades','joker','king of clubs','king of diamonds','king of hearts','king of spades','nine of clubs','nine of diamonds','nine of hearts','nine of spades','queen of clubs','queen of diamonds','queen of hearts','queen of spades','seven of clubs','seven of diamonds','seven of hearts','seven of spades','six of clubs','six of diamonds','six of hearts','six of spades','ten of clubs','ten of diamonds','ten of hearts','ten of spades','three of clubs','three of diamonds','three of hearts','three of spades','two of clubs','two of diamonds','two of hearts','two of spades']
uploaded_file = st.file_uploader(
    "Upload an image of a card:", type="jpg"
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(200,200))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("card.jpg")
    image1=image.smart_resize(image1,(200,200))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]
    image1 = Image.open("card.jpg")
    st.image(image1, caption="Uploaded Image", use_column_width=True)    
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )