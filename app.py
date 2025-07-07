import argparse
import os
from datetime import datetime

import gradio as gr
from gradio.themes.utils import colors, fonts
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


# --- Giao diện Tùy chỉnh ---
custom_theme = gr.themes.Soft(
    primary_hue=colors.blue,
    secondary_hue=colors.sky,
    neutral_hue=colors.slate,
    font=(fonts.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"),
)

custom_css = """
/* Custom CSS for VAA Graduation Project */
.gradio-container { background: #f0f4f8; }
#header-title { text-align: center; font-family: 'Poppins', sans-serif; color: #1e3a8a; font-size: 2.5rem; font-weight: 600; margin-bottom: 0px; }
#header-subtitle { text-align: center; font-family: 'Poppins', sans-serif; color: #3b82f6; font-size: 1.5rem; margin-top: 5px; margin-bottom: 10px; }
#header-academy { text-align: center; font-family: 'Poppins', sans-serif; color: #6b7280; font-size: 1.1rem; margin-bottom: 20px; }
.gradio-button.gr-button-primary { background: #2563eb; color: white; border: none; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); transition: all 0.2s ease-in-out; }
.gradio-button.gr-button-primary:hover { background: #1d4ed8; transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1); }
"""
# -------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--p2p_base_model_path",
        type=str,
        default="timbrooks/instruct-pix2pix", 
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--ip_base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting", 
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--p2p_resume_path",
        type=str,
        default="zhengchong/CatVTON-MaskFree",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--ip_resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.ip_resume_path)
# Pipeline
pipeline_p2p = CatVTONPix2PixPipeline(
    base_ckpt=args.p2p_base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix-48k-1024",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cpu'
)

# Pipeline
repo_path = snapshot_download(repo_id=args.ip_resume_path)  
pipeline = CatVTONPipeline(
    base_ckpt=args.ip_base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cpu'
)

# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cpu', 
)


def submit_function_p2p(
    person_image,
    cloth_image,
    num_inference_steps,
    guidance_scale,
    seed):
    person_image= person_image["background"]

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cpu').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # Inference
    try:
        result_image = pipeline_p2p(
            image=person_image,
            condition_image=cloth_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )[0]
    except Exception as e:
        raise gr.Error(
            "An error occurred. Please try again later: {}".format(e)
        )
    
    # Post-process
    save_result_image = image_grid([person_image, cloth_image, result_image], 1, 3)
    save_result_image.save(result_save_path)
    return result_image

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    # Map Vietnamese UI values to English backend values
    cloth_type_map = {
        "Áo": "upper",
        "Quần": "lower",
        "Toàn thân": "overall"
    }
    english_cloth_type = cloth_type_map.get(cloth_type)

    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cpu').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            english_cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image



def person_example_fn(image_path):
    return image_path

def app_gradio():
    with gr.Blocks(theme=custom_theme, css=custom_css, title="Đồ án Tốt nghiệp - Thử đồ ảo") as demo:
        # --- HEADER ---
        gr.Markdown(
            """
            <h1 id="header-title">ĐỒ ÁN TỐT NGHIỆP</h1>
            <h2 id="header-subtitle">Xây dựng Ứng dụng Thử đồ ảo (Virtual Try-On)</h2>
            <p id="header-academy">Học viện Hàng không Việt Nam</p>
            """
        )
        
        with gr.Tab("Thử đồ (có Mask)"):
            # --- Input Images Row ---
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    person_image = gr.ImageEditor(interactive=True, label="1. Tải ảnh người mẫu", type="filepath")
                with gr.Column(scale=1):
                    cloth_image = gr.Image(interactive=True, label="2. Tải ảnh trang phục", type="filepath")
            
            # --- Controls Row ---
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("##### 3. Tùy chỉnh (Tùy chọn)")
                    cloth_type = gr.Radio(
                        label="Loại trang phục (để tạo mask tự động nếu không vẽ tay)",
                        choices=["Áo", "Quần", "Toàn thân"],
                        value="Áo",
                    )
                    with gr.Accordion("Tùy chọn nâng cao", open=False):
                        num_inference_steps = gr.Slider(
                            label="Số bước xử lý", minimum=10, maximum=100, step=5, value=50
                        )
                        guidance_scale = gr.Slider(
                            label="Độ mạnh của CFG", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                        )
                        seed = gr.Slider(
                            label="Seed (ngẫu nhiên)", minimum=-1, maximum=10000, step=1, value=42
                        )
                        show_type = gr.Radio(
                            label="Kiểu hiển thị",
                            choices=["Chỉ kết quả", "Đầu vào & Kết quả", "Đầu vào & Mask & Kết quả"],
                            value="Đầu vào & Mask & Kết quả",
                        )
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown("##### 4. Bắt đầu")
                    submit = gr.Button("Tạo ảnh thử đồ", variant="primary", size="lg")
                    gr.Markdown(
                        '<center><span style="color: #FF0000">Lưu ý: Quá trình xử lý sẽ mất thời gian.</span></center>'
                    )

            # --- Output Row ---
            gr.Markdown("<hr>")
            with gr.Row():
                result_image = gr.Image(interactive=False, label="Kết quả")
            
            # Hidden image path for examples (if re-enabled)
            image_path = gr.Image(type="filepath", interactive=True, visible=False)
            image_path.change(person_example_fn, inputs=image_path, outputs=person_image)
            
            # Click Handler
            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )

        with gr.Tab("Thử đồ (không cần Mask)"):
            # --- Input Images Row ---
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    person_image_p2p = gr.ImageEditor(interactive=True, label="1. Tải ảnh người mẫu", type="filepath")
                with gr.Column(scale=1):
                    cloth_image_p2p = gr.Image(interactive=True, label="2. Tải ảnh trang phục", type="filepath")

            # --- Controls Row ---
            with gr.Row():
                with gr.Column(scale=3):
                     with gr.Accordion("Tùy chọn nâng cao", open=False):
                        num_inference_steps_p2p = gr.Slider(
                            label="Số bước xử lý", minimum=10, maximum=100, step=5, value=50
                        )
                        guidance_scale_p2p = gr.Slider(
                            label="Độ mạnh của CFG", minimum=0.0, maximum=7.5, step=0.5, value=2.5
                        )
                        seed_p2p = gr.Slider(
                            label="Seed (ngẫu nhiên)", minimum=-1, maximum=10000, step=1, value=42
                        )
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown("##### 3. Bắt đầu")
                    submit_p2p = gr.Button("Tạo ảnh thử đồ", variant="primary", size="lg")
                    gr.Markdown(
                        '<center><span style="color: #FF0000">Lưu ý: Quá trình xử lý sẽ mất thời gian.</span></center>'
                    )
            
            # --- Output Row ---
            gr.Markdown("<hr>")
            with gr.Row():
                result_image_p2p = gr.Image(interactive=False, label="Kết quả")
            
            # Hidden image path for examples (if re-enabled)
            image_path_p2p = gr.Image(type="filepath", interactive=True, visible=False)
            image_path_p2p.change(person_example_fn, inputs=image_path_p2p, outputs=person_image_p2p)

            # Click Handler
            submit_p2p.click(
                submit_function_p2p,
                [
                    person_image_p2p,
                    cloth_image_p2p,
                    num_inference_steps_p2p,
                    guidance_scale_p2p,
                    seed_p2p
                ],
                result_image_p2p,
            )
        
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()