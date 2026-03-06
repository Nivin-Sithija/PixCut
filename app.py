import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image, ImageFilter
import cv2
import io

# --- 1. Define Model Architecture ---
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights=None, # Loading from best_model.pt anyway
            in_channels=3,
            classes=1,
            activation=None
        )
        
    def forward(self, images, masks=None):
        logits = self.arc(images)
        return logits

# --- 2. Caching the model loading so it's fast ---
@st.cache_resource
def load_model(weights_path):
    model = SegmentationModel()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 3. Preprocessing function ---
def get_mask(model, image: Image.Image, size=(320, 320)):
    # Keep original dimensions to resize the mask back later
    original_size = image.size 
    
    # Resize and convert to tensor
    img_resized = image.convert("RGB").resize(size, Image.BILINEAR)
    img_arr = np.array(img_resized)
    img_tensor = np.transpose(img_arr, (2, 0, 1)).astype(np.float32)
    img_tensor = torch.Tensor(img_tensor) / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        logits_mask = model(img_tensor)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0
        
    # Squeeze out batch and channel dimensions finding the 2D mask
    mask_2d = pred_mask.squeeze().cpu().numpy()
    
    # Resize the mask back to original image size
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return mask_img

# --- 4. Main Streamlit App ---
def main():
    st.set_page_config(page_title="AI Creator Toolkit", page_icon="✨", layout="centered")
    
    st.title("✨ AI Creator Toolkit")
    st.write("Using your Custom Deep Learning Image Segmentation Model")
    
    st.markdown("---")
    
    # Load model
    try:
        model = load_model('best_model.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.sidebar.title("Toolkit Tools")
        mode = st.sidebar.radio("Choose an effect:", 
                                ["Show Mask Only", 
                                 "Extract Subject (Transparent BG)", 
                                 "Portrait Mode (Blur BG)", 
                                 "Pop Color (B&W BG)"])
        
        st.write("### Result")
        
        # Get predicted mask
        with st.spinner('Running deep learning model...'):
            mask = get_mask(model, original_image)
            
        if mode == "Show Mask Only":
            st.image(mask, caption="Predicted Segmentation Mask", use_container_width=True)
            
        elif mode == "Extract Subject (Transparent BG)":
            # Convert mask to alpha channel
            result = original_image.copy()
            result.putalpha(mask)
            st.image(result, caption="Extracted Subject", use_container_width=True)
            
            # Prepare download
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="Download Transparent PNG", data=byte_im, file_name="extracted_subject.png", mime="image/png")
            
        elif mode == "Portrait Mode (Blur BG)":
            # Blur the original image
            blurred_bg = original_image.filter(ImageFilter.GaussianBlur(15))
            
            # Composite original image over the blurred background using the mask
            result = Image.composite(original_image, blurred_bg, mask)
            st.image(result, caption="Portrait Mode", use_container_width=True)

        elif mode == "Pop Color (B&W BG)":
            # Convert original image to grayscale, then back to RGB to retain channels
            bw_bg = original_image.convert("L").convert("RGB")
            
            # Composite original image over the B&W background using the mask
            result = Image.composite(original_image, bw_bg, mask)
            st.image(result, caption="Color Pop", use_container_width=True)

if __name__ == '__main__':
    main()
