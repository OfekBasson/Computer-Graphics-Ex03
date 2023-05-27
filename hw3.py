from IPython.core.prefilter import is_shadowed
from helper_classes import *
import matplotlib.pyplot as plt

EPSILON = 1e-5

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            obj, distance = ray.nearest_intersected_object(objects)
            hit_point = ray.origin + distance*ray.direction
            color = get_color(lights, objects, ray, obj, hit_point, distance, 1, max_depth, ambient)
     
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

def get_color(lights, objects, ray, obj, hit_point, min_distance, current_depth, max_depth, ambient):
  normal_of_obj_at_hit_point = obj.compute_normal(hit_point)
  hit_point += EPSILON*normal_of_obj_at_hit_point
  rgb_color = calc_ambient_color(obj, ambient)
  for light in lights:
    ray_to_light = light.get_light_ray(hit_point)
    if not is_light_blocked(light, obj, hit_point, objects, ray_to_light):
      rgb_color += calc_diffuse_color(light, obj, ray_to_light, normal_of_obj_at_hit_point, hit_point) + calc_specular_color(light, obj, ray, hit_point, ray_to_light, normal_of_obj_at_hit_point)

  if current_depth < max_depth:
    reflective_ray = get_reflective_ray(ray, normal_of_obj_at_hit_point, hit_point)
    reflective_ray_nearest_obj, reflective_ray_nearest_distance = reflective_ray.nearest_intersected_object(objects)
    new_hit_point = reflective_ray.origin + reflective_ray_nearest_distance*reflective_ray.direction
    if reflective_ray_nearest_obj is not None:
      rgb_color += obj.reflection * get_color(lights, objects, reflective_ray, reflective_ray_nearest_obj, new_hit_point, reflective_ray_nearest_distance, current_depth + 1, max_depth, ambient)

  return rgb_color

def get_reflective_ray(ray, normal_of_obj_at_hit_point, hit_point):
  direction = reflected(ray, normal_of_obj_at_hit_point, hit_point)
  return Ray(hit_point, direction)

def is_light_blocked(light, obj, hit_point, objects, ray_to_light):
  ray_from_light_to_hit_point = get_ray_from_light_to_obj(light, hit_point)
  nearest_object_to_light, t = ray_from_light_to_hit_point.nearest_intersected_object(objects)
  if nearest_object_to_light is not obj and nearest_object_to_light is not None:
    return True
  return False

def get_ray_from_light_to_obj(light, hit_point):
  if isinstance(light, DirectionalLight):
    origin = hit_point + 10000*light.direction
    return Ray(origin, -light.direction)
  elif isinstance(light, SpotLight) or isinstance(light, PointLight):
    return Ray(light.position, normalize(hit_point - light.position))
  else:
    raise Exception("Not a light")

def calc_ambient_color(obj, ambient):
  return (obj.ambient * ambient).astype('float64')

def calc_diffuse_color(light, obj, ray_to_light, normal, hit_point):
  diffuse_color = obj.diffuse * light.get_intensity(hit_point) * np.dot(normal, ray_to_light.direction)
  return diffuse_color

def calc_specular_color(light, obj, ray, hit_point, ray_to_light, normal):
  reflected_light_ray = reflected(ray_to_light, normal, hit_point)
  specular_color = obj.specular * light.get_intensity(hit_point) * np.power(np.dot(-ray.direction, reflected_light_ray), obj.shininess)
  return specular_color

def construct_reflective_ray():
  #TODO
  pass


# Write your own objects and lights
def your_own_scene():
    stomach = Cuboid(
      [-1, 3, -4],
      [-1, -1, -4],
      [1, -1, -6],
      [1, 3, -6],
      [0, -1, -7], 
      [0, 3, -7]
    )
    
    right_hand = Cuboid(
      [-3, 3, -2],
      [-3, 1.6, -2],
      [-1, 1.6, -4],
      [-1, 3, -4],
      [-2, 1.6, -5], 
      [-2, 3, -5]   
    )

    left_hand = Cuboid(
      [1, 3, -6],
      [1, 1.6, -6],
      [3, 1.6, -8],
      [3, 3, -8],
      [2, 1.6, -9], 
      [2, 3, -9]   
    )

    left_pants = Cuboid(
      [0.2, -1, -5.2],
      [0.2, -3, -5.2],
      [1, -3, -6],
      [1, -1, -6],
      [0, -3, -7], 
      [0, -1, -7]
    )

    right_pants = Cuboid(
      [-1, -1, -4],
      [-1, -3, -4],
      [-0.2, -3, -4.8],
      [-0.2, -1, -4.8],
      [-1.2, -3, -5.8], 
      [-1.2, -1, -5.8]
    )

    left_leg = Cuboid(
      [0.2, -3, -5.2],
      [0.2, -5, -5.2],
      [1, -5, -6],
      [1, -3, -6],
      [0, -5, -7], 
      [0, -3, -7]
    )

    right_leg = Cuboid(
      [-1, -3, -4],
      [-1, -5, -4],
      [-0.2, -5, -4.8],
      [-0.2, -3, -4.8],
      [-1.2, -5, -5.8], 
      [-1.2, -3, -5.8]
    )

    head = Sphere([-0.5, 4.5, -5.5],1.5)

    stomach.set_material([1, 0, 0.5], [1, 0, 0.5], [0, 0, 0], 100, 0.5)
    stomach.apply_materials_to_faces()

    right_hand.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
    right_hand.apply_materials_to_faces()

    left_hand.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
    left_hand.apply_materials_to_faces()

    right_pants.set_material([1, 0, 0.5], [1, 0, 0.5], [0, 0, 0], 100, 0.5)
    right_pants.apply_materials_to_faces()

    left_pants.set_material([1, 0, 0.5], [1, 0, 0.5], [0, 0, 0], 100, 0.5)
    left_pants.apply_materials_to_faces()

    right_leg.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
    right_leg.apply_materials_to_faces()

    left_leg.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
    left_leg.apply_materials_to_faces()

    head.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)

    background = Plane([0,0,1], [0,0,-10])
    background.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 100, 0.5)


    objects = [stomach, right_hand, left_hand, left_pants, right_pants, right_leg, left_leg, head, background]

    point_light = PointLight(intensity = np.array([1, 1, 1]),position=np.array([4,1,1]),kc=0.1,kl=0.1,kq=0.1)
    directional_light = DirectionalLight(intensity=np.array([1, 1, 1]), direction=np.array([0,1,1]))

    lights = [point_light, directional_light]
    camera = np.array([0,0,1])

    return camera, lights, objects
