# CS488 Final Project

## Compilation

Compile and run with cmake and g++.

## Specification

This project now uses path tracing which is much slower, so the image quality increases each frame as each frame adds more samples per pixel.

### Path Tracing

Images are now rendered by a path tracing algorithm. The path tracer uses Monte Carlo integration, russian roulette, importance sampling, and parallelization to improve performance. The path tracer renders images with color-bleeding and global illumination which makes rendered images look more real.

```
TODO: images that demonstrate each of these things
```

### Lambertian IBL

The path tracer also implements IBL for all material types now, including Lambertian.

```
TODO: images that demonstrate each of these things
```

### Area Lights

The path tracer uses area lights instead of point lights which are more realistic to lights we see in real life. The path tracer renders soft shadows, which look much more real than the shadows that the raytracer had.

```
TODO: images that demonstrate each of these things
```

### Fresnel

### Microfacets

### Alpha Blending

### Kd-tree

A Surface-Area Heuristic Kd-tree is implemented and can be used instead of the SAH BVH. To control whether the Kd-tree is used, you can comment/uncomment the line `#define KD_TREE`.

teapot.obj timing:

SAH BVH: 0.4s per frame

SAH Kd-tree: 0.6s per frame

The Kd-tree is slightly slower but works perfectly.

## References

### Path Tracing, Area Lights, IBL, Fresnel

requation, smallpt, and path lecture slides

https://www.kevinbeason.com/smallpt/

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
