import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO

# 下载图像
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# 显示图像
def show_images(images, rows, cols):
    assert len(images) <= rows * cols
    w, h = images[0].size
    grid_img = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid_img.paste(img, box=(i % cols * w, i // cols * h))
    return grid_img

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = "img.png"
init_image = download_image(url)
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

grid_img = show_images([init_image, images[0]], 1, 2)
grid_img.save("fantasy_landscape.png")