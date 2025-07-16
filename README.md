# AgTactileEngine
Image/texture to 3D tactile models. 

## How To Use

``` python

from agtactileengine2 import AgTactileEngine

images = [
    'bark.jpg',
    'brick.png',
    'grass.png',
]

DepthEngine = AgTactileEngine()

for image in images:
    image_path = image
    cropped_output_path = f'./output/cropped_{image}'
    smooth_output_path = f'./output/smooth_{image}'
    supersmooth_output_path = f'./output/supersmooth_{image}'
    
    DepthEngine.crop_image_center(image_path, cropped_output_path, size=(512, 512))

    depth_output_path = f'./output/depth_{image}'
    DepthEngine.calculate_depth(cropped_output_path, depth_output_path, smooth_output_path=smooth_output_path, supersmooth_output_path=supersmooth_output_path, supersmooth_strength=5)

    DepthEngine.generate_3d_stl(supersmooth_output_path, output_path=f'./output/model_{image}.stl')

```
