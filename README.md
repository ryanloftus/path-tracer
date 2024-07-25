# cs488

## a2

### Compilation

Compile and run with cmake and g++.

### Specification

Task 1 uses Verlet Integration. 

Task 2 uses particle collision for position based particle motion as described in the lecture slides.

Task 3 uses vector projection to project the position onto the sphere.

Task 4 adds a forces parameter to the Particle::step() function that contains the inter-particle gravitational forces computed by the ParticleSystem. Tested using particle mass of 4e8f.

Task 5 uses the position based response to collisions and conserves momentum by altering the prevPosition.
Task 5 solves multiple collisions at the same time by iterating over each pair of particles several times to resolve simultaneous collisions one at a time.

### Extras

For Extra 1, I implemented Barnes-Hut to speed up particle-particle gravity computations. Extra 1 can be enabled by uncommenting the line `#define A3_BONUS_1`.

The performance improvement for running with 5000 single-triangle particles on my laptop was:

0.018 seconds with enhancement.
2.917 without enhancement.

The time was measured by from the start of `computeAccumulatedForces()` to the end of `computeAccumulatedForces()` for one frame. Measurements given take the average over the first 10 frames.

## References

https://www.kevinbeason.com/smallpt/
