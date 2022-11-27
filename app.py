from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

W = 768
H = 768
N = 10
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.enable_attention_slicing()
prompt = "schnauzer chihuahua mix painted by Alphonse Mucha, natural lighting, path traced, highly detailed, high quality, digital painting, by don bluth and ross tran and studio ghibli and alphonse mucha, artgerm "
filename_base = prompt.replace(',', '-').replace(':','').replace(' ', '_')[:200]

for i in range(N):
    images = pipe(prompt, height=W, width=H).images
    images[0].save(f'{filename_base}_{i}.png')
