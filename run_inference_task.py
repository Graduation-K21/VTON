import argparse
import os
from PIL import Image
import torch
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker

def main():
    parser = argparse.ArgumentParser(description='Run virtual try-on inference')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID')
    parser.add_argument('--person_image', type=str, required=True, help='Path to person image')
    parser.add_argument('--cloth_image', type=str, required=True, help='Path to cloth image')
    args = parser.parse_args()

    # Khởi tạo pipeline và masker
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt="Models/CatVTON",
        attn_ckpt_version="mix",
        device=device,
        skip_safety_check=True  # Bỏ qua kiểm tra an toàn để tăng tốc độ
    )
    masker = AutoMasker(
        densepose_ckpt='Models/DensePose',
        schp_ckpt='Models/SCHP',
        device=device
    )

    try:
        # Đọc ảnh đầu vào
        person_image = Image.open(args.person_image).convert('RGB')
        cloth_image = Image.open(args.cloth_image).convert('RGB')
        
        # Tạo mask cho người
        person_masks = masker.preprocess_image(person_image)
        person_mask = masker.cloth_agnostic_mask(
            densepose_mask=person_masks['densepose'],
            schp_lip_mask=person_masks['schp_lip'],
            schp_atr_mask=person_masks['schp_atr'],
            part='upper'  # hoặc 'lower' tùy vào loại quần áo
        )

        # Chạy inference
        result_images = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=person_mask,
            num_inference_steps=30  # Giảm số bước để tăng tốc độ
        )
        result_image = result_images[0]  # Lấy ảnh đầu tiên từ batch

        # Lưu kết quả
        result_path = os.path.join('results', f'{args.job_id}.png')
        result_image.save(result_path)

    except Exception as e:
        print(f"Error processing job {args.job_id}: {e}")
        raise

if __name__ == '__main__':
    main() 