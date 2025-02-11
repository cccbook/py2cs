# https://huggingface.co/docs/diffusers/main/en/index

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

print(pipeline)

# pipeline.to("cuda")

image = pipeline("An image of a squirrel in Picasso style").images[0]
# image = pipeline("A squirrel walk on water").images[0]
# image
image.save("image_of_squirrel_painting.png")

