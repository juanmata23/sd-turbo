import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
    # FloatXX 32 o 16
    # Me indica la cantidad de decimales de precision
    # con el que el modelo creara la imagen
    # Si mi pc me genera una imagen negra, cambio a 16
)

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - Mac com procesadores MX
# xpu - Grafica Intel IRIS

prompt = input("Escribe el prompt de la imagen: ")

print("Generando la imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=("Bad anatomy, Bad proportions, Bad quality, Blurry, Collage, Cropped, Deformed, Dehydrated, Disconnected limbs, Disfigured, Disgusting, Error, Extra arms, Extra hands, Extra limbs, Fused fingers, Grainy, Gross proportions, Jpeg, Jpeg artifacts, Long neck, Low quality, Low res, Malformed limbs, Missing arms, Missing fingers, Mutated, Mutated hands, Mutated limbs, Out of focus, Out of frame, Picture frame, Pixel, Pixelated, Poorly drawn face, Poorly drawn hands, Signature, Text, Ugly,2D, 3D, 3D Rendering, Anime, Artstation, Artwork, Bad photography, Bad picture, CGI, Cartoons, Cinema 4D, Deviant art, Drawing, Illustration, Octane render, Oil painting, Painting, Render, Sketch,ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"),
    num_inference_steps=10,  # Numero de veces que creara y corregira la imagen
    guidance_scale=2,  # Que tan fiel sera al prompt
).images[0]

imagen.save("imagen.png")
print("Imagen guardada !")



