import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

# Set page config
st.set_page_config(
    page_title="Face Detection & Background Removal",
    page_icon=":camera:",
    layout="wide"
)

# App title and description
st.title("📸 Face Detection & Background Removal")
st.markdown("""
Upload a photo to detect faces and remove the background. 
The processed image can be downloaded as a PNG with transparent background.
""")

# Sidebar controls
st.sidebar.header("Settings")
detection_scale = st.sidebar.slider("Detection Scale Factor", 1.01, 1.5, 1.1, step=0.01)
min_neighbors = st.sidebar.slider("Detection Min Neighbors", 1, 10, 5)
show_rectangles = st.sidebar.checkbox("Show Face Rectangles", True)

# Upload image
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="uploader")

def detect_faces(image):
    """Detect faces using OpenCV Haar Cascade"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=detection_scale, 
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    return faces

def draw_faces(image, faces):
    """Draw rectangles around detected faces"""
    img_with_faces = image.copy()
    for (x, y, w, h) in faces:
        if show_rectangles:
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
    return img_with_faces

def remove_background(image):
    """Remove background using rembg"""
    return remove(image)

if uploaded_file is not None:
    try:
        # Read and prepare image
        image = Image.open(uploaded_file)
        st.sidebar.success("Image uploaded successfully!")
        
        # Convert to OpenCV format
        img_cv = np.array(image.convert('RGB'))
        img_cv = img_cv[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        # Display original image
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Detect faces
        with st.spinner('Detecting faces...'):
            faces = detect_faces(img_cv)
        
        # Draw faces on image
        img_with_faces = draw_faces(img_cv, faces)
        
        # Convert back to PIL format for display
        img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)
        img_with_faces_pil = Image.fromarray(img_with_faces_rgb)
        
        # Display image with faces
        with col2:
            st.subheader(f"Detected {len(faces)} Faces")
            st.image(img_with_faces_pil, use_container_width=True)
        
        # Background removal
        st.subheader("Background Removal")
        with st.spinner('Removing background...'):
            # Use original image for background removal
            bg_removed = remove_background(image)
            
            # Convert to RGBA to preserve transparency
            bg_removed = bg_removed.convert("RGBA")
            
            # Display result
            st.image(bg_removed, caption="Background Removed", use_container_width=True)
            
            # Create download button
            buf = io.BytesIO()
            bg_removed.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Background-Free Image",
                data=byte_im,
                file_name="background_free.png",
                mime="image/png",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("👆 Please upload an image to get started")

# Add some tips
st.markdown("---")
st.subheader("Usage Tips")
st.markdown("""
- For best results, use photos with clear front-facing faces
- Adjust detection parameters in the sidebar if faces aren't detected
- Background removal works best with human subjects
- Larger images will take longer to process
- The download will be a PNG with transparent background
""")