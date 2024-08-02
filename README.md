# CS488 Final Project

## Compilation

Compile and run with cmake and g++.

```
$ make
...
$ ./CS488 <path_to_obj_file> <optional_path_to_hdr_image>
```

## Specification

This project now uses path tracing which is much slower, so the image quality increases each frame as each frame adds more samples per pixel. This project extends the existing cs488 code and adds new features: path tracing, area lights, alpha blending, lambertian IBL, kd-trees, Fresnel reflection, and microfacet models.

Translucent objects can be rendered by adding `Op X` as a separate line in the mtl file under the material that should be translucent. 0 <= X <= 1 and X is the opacity of the material (1.0 means the material is opaque and 0.0 means the material is transparent).

For metals, a roughness parameter `Ro X` can be specified in the mtl file where 0 <= X <= 1 and X is the roughness of the metal. An index of refraction parameter `Ri X` can also be provided for metals.

For area lights, an `Ke R G B` where R, G, B are the red, green, and blue components of the light.

To control whether the Kd-tree is used, you can comment/uncomment the line `#define KD_TREE`.

The constant value `SAMPLES_PER_PIXEL` determines the number of samples the path tracer takes for each pixel in one frame. The line `#define PROGRESSIVE_PATHTRACING` can be commented-out to turn off the progressive path tracing logic so that each frame starts from a clear buffer and the quality does not improve each frame.

Each frame, the program outputs two lines. One is the total number of samples per pixel that were used for that frame, the other is the amount of time, in seconds, that it took to render the frame.

## Implementation

### Path Tracing

Images are now rendered by a path tracing algorithm. The path tracer uses Monte Carlo integration, russian roulette, importance sampling, and parallelization to improve performance. Samples are taken progressively so each frame adds 10 more samples to the image.

### Area Lights

Instead of iterating over point lights to find light contribution for a point on a surface, we use an emmittance parameter that is positive for luminous surfaces and is added to the pixel color of hits on those surfaces.

### Lambertian IBL

The path tracer also implements IBL for all material types now, including Lambertian. When a ray hits a Lambertian surface, we reflect a random ray off of it, if that ray doesn't hit anything, we map a value from the environment map (if there is one).

### Fresnel Reflection

Fresnel reflections for glass is implemented using the Fresnel formula that was presented in lecture. The value of the Fresnel equation is the probability that we trace the reflected ray vs. the refracted ray.

### Alpha Blending

When a translucent material is hit, we use the formula:
```
Opacity * materialBRDF + (1 - Opacity) * nextMaterialBRDF
```
where the `nextMaterialBRDF` is found by tracing a ray going in the same direction as the current ray, past the current surface.

### Microfacet Models

The Cook-Torrance model is used to simulate the roughness of metal surfaces. For the Cook-Torrance implementation, the Schlick approximation for Fresnel is used and GGX approximation for the normal distribution is used. In order to perform Cook-Torrance, we need a random reflected ray in the region of the hemisphere of the surface between viewDir and the surface, with the normal inside. For this, code from rorydriscoll's raytracer is referenced to get a half-vector that determines a ray in the desired region. We then apply the Cook-Torrance model.

### Kd-Tree

A Surface-Area Heuristic Kd-tree is implemented and can be used instead of the SAH BVH.

teapot.obj timing:

SAH BVH: 0.5s per frame

SAH Kd-tree: 0.4s per frame

cornellbox.obj timing:

SAH BVH: 2.8s per frame

SAH Kd-tree: 2.4s per frame

The Kd-tree is slightly faster and works perfectly.

## Objectives

- Soft shadows and color bleeding (path tracing)
- Global illumination (path tracing)
- Realistic glass (Fresnel reflection)
- Realistic metals (microfacets)
- Translucent materials (alpha blending)
- IBL for lambertian materials
- Kd-tree for ray-triangle intersection tests
- Area lights

## References

### Path Tracing, Area Lights

requation, smallpt, and path lecture slides

https://www.kevinbeason.com/smallpt/

### Fresnel, IBL

shading lecture slides.

### Microfacets

https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf

https://graphicscompendium.com/gamedev/15-pbr

https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

https://graphics.pixar.com/library/ReflectanceModel/paper.pdf

https://github.com/pboechat/cook_torrance/blob/master/application/shaders/cook_torrance_colored.fs.glsl

https://github.com/rorydriscoll/RayTracer

### Alpha Blending

https://www.phatcode.net/articles.php?id=233

### Kd-Tree

acceleration1 and acceleration2 lecture slides.

https://ieeexplore.ieee.org/document/4061547
