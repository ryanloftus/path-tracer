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

A Surface-Area Heuristic Kd-tree is implemented and can be used instead of the SAH BVH. 

teapot.obj timing:

SAH BVH: 0.4s per frame

SAH KdTree: 0.8s per frame

## References

### Path Tracing

https://www.kevinbeason.com/smallpt/

### Microfacets

https://github.com/pboechat/cook_torrance/blob/master/application/shaders/cook_torrance_colored.fs.glsl

https://github.com/rorydriscoll/RayTracer
