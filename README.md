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


