from PIL import Image, ImageFilter
import os
import numpy as np
from stl import mesh

class AgTactileEngine:
    def __init__(self):
        print("AgTactileEngine initialized.")

    def crop_image_center(self, image_path: str, output_path: str, size: tuple = (500, 500)):
        try:
            img = Image.open(image_path)
            print(f"Opened image: {image_path}")
            img_width, img_height = img.size
            crop_width, crop_height = size

            if img_width > img_height:
                img = img.resize((int(img_width * crop_height / img_height), crop_height), Image.Resampling.LANCZOS)
            else:
                img = img.resize((crop_width, int(img_height * crop_width / img_width)), Image.Resampling.LANCZOS)

            img_width, img_height = img.size
            left = (img_width - crop_width) / 2
            top = (img_height - crop_height) / 2
            right = (img_width + crop_width) / 2
            bottom = (img_height + crop_height) / 2

            cropped_img = img.crop((left, top, right, bottom))
            print(f"Image cropped and resized to {size} at center.")
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            cropped_img.save(output_path)
            print(f"Cropped image saved to: {output_path}")

        except FileNotFoundError:
            print(f"Error: Input image file not found at {image_path}.")
        except IOError as e:
            print(f"Error saving cropped image to {output_path}: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred during cropping: {e}")

    def calculate_depth(self, image_path: str, output_path: str, smooth_output_path: str, supersmooth_output_path: str = None, supersmooth_strength: int = 50):
        try:
            img = Image.open(image_path)
            print(f"Opened image for depth estimation: {image_path}")
            
            depth_map = img.convert("L")
            print("Monochrome depth map estimated.")
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            depth_map.save(output_path)
            print(f"Depth map saved to: {output_path}")

            smooth_depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=5))
            print("Smooth depth map created by applying a Gaussian blur.")
            smooth_output_dir = os.path.dirname(smooth_output_path)
            if smooth_output_dir and not os.path.exists(smooth_output_dir):
                os.makedirs(smooth_output_dir, exist_ok=True)
            smooth_depth_map.save(smooth_output_path)
            print(f"Smooth depth map saved to: {smooth_output_path}")

            if supersmooth_output_path:
                supersmooth_strength = max(2, min(100, supersmooth_strength))
                supersmooth_radius = supersmooth_strength / 10.0
                supersmooth_depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=supersmooth_radius))
                print(f"Super-smooth depth map created with strength {supersmooth_strength}.")
                supersmooth_output_dir = os.path.dirname(supersmooth_output_path)
                if supersmooth_output_dir and not os.path.exists(supersmooth_output_dir):
                    os.makedirs(supersmooth_output_dir, exist_ok=True)
                supersmooth_depth_map.save(supersmooth_output_path)
                print(f"Super-smooth depth map saved to: {supersmooth_output_path}")

        except FileNotFoundError:
            print(f"Error: Input image file not found at {image_path}.")
        except IOError as e:
            print(f"Error saving image: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred during depth estimation: {e}")

    def generate_3d_stl(self, image_path: str, output_path: str, height_scale: float = 10.0, base_height: float = 2.0):
        try:
            img = Image.open(image_path).convert("L")
            img_array = np.array(img, dtype=np.float32)

            height, width = img_array.shape

            vertices = np.zeros((height, width, 3))
            for y in range(height):
                for x in range(width):
                    z = (img_array[y, x] / 255.0) * height_scale
                    vertices[y, x] = [x, y, z]

            surface_triangles_count = (width - 1) * (height - 1) * 2
            surface_data = np.zeros(surface_triangles_count, dtype=mesh.Mesh.dtype)

            face_idx = 0
            for y in range(height - 1):
                for x in range(width - 1):
                    surface_data['vectors'][face_idx] = [
                        vertices[y, x],
                        vertices[y + 1, x],
                        vertices[y, x + 1]
                    ]
                    face_idx += 1
                    surface_data['vectors'][face_idx] = [
                        vertices[y + 1, x],
                        vertices[y + 1, x + 1],
                        vertices[y, x + 1]
                    ]
                    face_idx += 1

            all_data = [surface_data]

            if base_height > 0:
                base_data = np.zeros(2, dtype=mesh.Mesh.dtype)
                base_z = -base_height
                base_data['vectors'][0] = [[0, 0, base_z], [0, height - 1, base_z], [width - 1, height - 1, base_z]]
                base_data['vectors'][1] = [[0, 0, base_z], [width - 1, height - 1, base_z], [width - 1, 0, base_z]]
                all_data.append(base_data)
            
            if base_height > 0:
                side_data = np.zeros((width * 2 + height * 2 - 4) * 2, dtype=mesh.Mesh.dtype)
                side_idx = 0
                base_z = -base_height

                for x in range(width - 1):
                    p1 = [x, 0, vertices[0, x, 2]]
                    p2 = [x + 1, 0, vertices[0, x + 1, 2]]
                    p3 = [x, 0, base_z]
                    p4 = [x + 1, 0, base_z]
                    side_data['vectors'][side_idx] = [p1, p3, p4]
                    side_idx += 1
                    side_data['vectors'][side_idx] = [p1, p4, p2]
                    side_idx += 1

                for x in range(width - 1):
                    p1 = [x, height - 1, vertices[height - 1, x, 2]]
                    p2 = [x + 1, height - 1, vertices[height - 1, x + 1, 2]]
                    p3 = [x, height - 1, base_z]
                    p4 = [x + 1, height - 1, base_z]
                    side_data['vectors'][side_idx] = [p1, p4, p3]
                    side_idx += 1
                    side_data['vectors'][side_idx] = [p1, p2, p4]
                    side_idx += 1

                for y in range(height - 1):
                    p1 = [0, y, vertices[y, 0, 2]]
                    p2 = [0, y + 1, vertices[y + 1, 0, 2]]
                    p3 = [0, y, base_z]
                    p4 = [0, y + 1, base_z]
                    side_data['vectors'][side_idx] = [p1, p3, p4]
                    side_idx += 1
                    side_data['vectors'][side_idx] = [p1, p4, p2]
                    side_idx += 1

                for y in range(height - 1):
                    p1 = [width - 1, y, vertices[y, width - 1, 2]]
                    p2 = [width - 1, y + 1, vertices[y + 1, width - 1, 2]]
                    p3 = [width - 1, y, base_z]
                    p4 = [width - 1, y + 1, base_z]
                    side_data['vectors'][side_idx] = [p1, p4, p3]
                    side_idx += 1
                    side_data['vectors'][side_idx] = [p1, p2, p4]
                    side_idx += 1
                
                all_data.append(side_data)

            combined_data = np.concatenate(all_data)
            final_mesh = mesh.Mesh(combined_data, remove_empty_areas=False)

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            final_mesh.save(output_path)
            print(f"3D STL object saved to: {output_path}")

        except FileNotFoundError:
            print(f"Error: Input image file not found at {image_path}.")
        except IOError as e:
            print(f"Error saving STL file to {output_path}: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred during STL generation: {e}")