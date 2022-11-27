from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

W = 512
H = 512
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# pipe.enable_attention_slicing()
prompt = "girl wearing cyberpunk intricate streetwear riding dirt bike, beautiful, detailed portrait, cell shaded, 4 k, concept art, by wlop, ilya kuvshinov, artgerm, krenz cushart, greg rutkowski, pixiv. cinematic dramatic atmosphere, sharp focus, volumetric lighting, cinematic lighting, studio quality"
filename = prompt.replace(',', '-').replace(' ', '_')[:200] + '.png'

images = pipe(prompt, height=W, width=H).images
    
images[0].save(filename)
