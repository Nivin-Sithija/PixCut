import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image, ImageFilter
import cv2
import io

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
    def forward(self, images, masks=None):
        logits = self.arc(images)
        return logits

@st.cache_resource #cache model
def load_model(weights_path):
    model = SegmentationModel()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_mask(model, image: Image.Image, size=(320, 320)):

    original_size = image.size 
    img_resized = image.convert("RGB").resize(size, Image.BILINEAR)
    img_arr = np.array(img_resized)
    img_tensor = np.transpose(img_arr, (2, 0, 1)).astype(np.float32)
    img_tensor = torch.Tensor(img_tensor) / 255.0 #normalize 0-1
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad(): #for inf fast
        logits_mask = model(img_tensor)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0 #0|1

    mask_2d = pred_mask.squeeze().cpu().numpy()
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return mask_img

# App 
def main():
    st.set_page_config(page_title="Image Segmentation", layout="wide")

    st.markdown("""
        <style>
        .title-h1 {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 800;
            color: #2D3748;
            padding-bottom: 0.5rem;
            margin-top: 1rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #A78BFA;
            background-color: #F5F3FF;
            border-radius: 8px;
            padding: 3rem;
        }
        [data-testid="stFileUploader"] label {
            display: none;
        }
        .upload-text {
            text-align: center;
            font-size: 0.9rem;
            color: #6B7280;
            margin-top: -1rem;
            margin-bottom: 2rem;
        }
        </style>
        <div class="title-h1">Image Segmentation</div>
    """, unsafe_allow_html=True) #html use 
    
    try:
        model = load_model('best_model.pt')
    except Exception as e:
        st.error(f"Error: {e}")
        return

    col_controls, col_original, col_result = st.columns([1, 1.2, 1.2], gap="large")
    
    with col_controls:
        uploaded_file = st.file_uploader("Choose Files", type=["jpg", "jpeg", "png"])
        
        with st.expander("Advanced settings", expanded=True):
            mode = st.radio("Choose an effect:", 
                            ["Show Mask Only", 
                             "Transparent BG", 
                             "Portrait Mode (Blur BG)", 
                             "B&W BG"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        with col_original:
            st.markdown("<h3 style='text-align: center; margin-top: 0rem;'>Original</h3>", unsafe_allow_html=True)
            st.image(original_image,use_container_width=True)     
        with col_result:
            st.markdown("<h3 style='text-align: center; margin-top: 0rem;'>Result</h3>", unsafe_allow_html=True)
            
            with st.spinner('Processing your image...'):
                mask = get_mask(model, original_image)

            #my modes    
            if mode == "Show Mask Only":
                result = mask
                caption = "Only Mask"
                download_name = "mask.png"
                
            elif mode == "Transparent BG":
                result = original_image.copy()
                result.putalpha(mask)#removebg
                caption = "Transparent BG"
                download_name = "transparent_bg.png"
                
            elif mode == "Portrait Mode (Blur BG)":
                blurred_bg = original_image.filter(ImageFilter.GaussianBlur(15))
                result = Image.composite(original_image, blurred_bg, mask)
                caption = "Portrait Mode"
                download_name = "portrait_mode.png"
    
            elif mode == "B&W BG":
                bw_bg = original_image.convert("L").convert("RGB")
                result = Image.composite(original_image, bw_bg, mask)
                caption = "B&W BG"
                download_name = "bw_bg.png"
                
            st.image(result, caption=caption, use_container_width=True)
            
            buf = io.BytesIO()#temporary buffer for data
            result.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            d_col1, d_col2, d_col3 = st.columns([1, 2, 1])
            with d_col2:
                st.download_button(label="Download Image", data=byte_im, file_name=download_name, mime="image/png", use_container_width=True)

if __name__ == '__main__':
    main()
