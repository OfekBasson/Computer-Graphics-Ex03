import numpy as np

# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis, hit_point):
    axis = normalize(axis)
    v = vector.direction - 2*np.dot(vector.direction, axis)*axis
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from a point to the light source 
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return 1000000

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source 
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))

class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source 
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        f_att = (self.kc + self.kl*d + self.kq * (d**2))
        intensity = (self.intensity * np.dot(self.get_light_ray(intersection).direction, self.direction)) / f_att
        return intensity.astype('float64')


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf

        for obj in objects:
          intersection_t_and_object = obj.intersect(self)
          if intersection_t_and_object is not None:
            t, obj = intersection_t_and_object
            if t < min_distance:
              min_distance, nearest_object = t, obj
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess/2
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None

    #Calculates the normal that starts at the point and goes out
    def compute_normal(self, hit_point):
      return normalize(self.normal)
      


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal(self.abcd[0])

    #Calculates the normal that starts at the point and goes out
    def compute_normal(self, hit_point):
        first_vector = self.abcd[1] - self.abcd[0]
        second_vector = self.abcd[2] - self.abcd[1]
        n = normalize(np.cross(first_vector, second_vector))
        return n

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        v = self.abcd[0] - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if (t > 0) and self.intersection_is_inside_rectangle(t, ray):
            return t, self
        else:
            return None

    def intersection_is_inside_rectangle(self, t, ray):
        hit_point = ray.origin + t*ray.direction
        for i in range(4):
          first_vector = self.abcd[i] - hit_point
          second_vector = self.abcd[(i + 1) % 4] - hit_point
          if np.dot(np.cross(first_vector, second_vector), self.normal) < 0:
            return False
        return True

class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        g = np.array(a) + np.array(f) - np.array(d)
        h = np.array(b) + np.array(f) - np.array(d)
        
        A = Rectangle(a, b, c, d)
        B = Rectangle(d, c, e, f)
        C = Rectangle(f, e, h, g)
        D = Rectangle(g, h, b, a)
        E = Rectangle(g, a, d, f)
        F = Rectangle(h, b, c, e)

        self.face_list = [A,B,C,D,E,F]

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        t = np.inf
        nearest_face = None
        for face in self.face_list:
          distance_and_intersection = face.intersect(ray)
          if distance_and_intersection is not None:
            distance, face_examined = distance_and_intersection
            if distance < t:
              t = distance
              nearest_face = face_examined
        if t == np.inf:
          return None
        return t, nearest_face

    #Calculates the normal that starts at the point and goes out
    def compute_normal(self, point):
      for face in self.face_list:
        point_on_the_face = face.abcd[0]
        vector_to_check_if_on_the_face = point_on_the_face - point
        if np.dot(vector_to_check_if_on_the_face, face.normal) == 0:
          return face.normal
        raise Exception("Point is not on the cuboid")
      

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return min(t1, t2), self
        elif t1 > 0 or t2 > 0:
            return max(t1, t2), self
        return None

    #Calculates the normal that starts at the point and goes out
    def compute_normal(self, point):
        norm = normalize(point - self.center)
        plane = Plane(norm, point)
        return plane.compute_normal(point)
