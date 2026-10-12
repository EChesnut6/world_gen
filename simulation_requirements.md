# Simulation Requirements (Draft)

## Purpose

Add a small, repeatable time-evolution mode to the world generator. The first version is a **kinematic model**: plate motion is prescribed, and the simulation advances crust and terrain through time. It does not solve mantle convection or calculate plate forces. Its purpose is to make motion and later terrain rules measurable and reproducible.

## Current starting point

- The world is represented on a spherical triangle mesh.
- Plates have angular velocity vectors; local rigid-plate velocity is calculated as `omega × position`.
- Boundary segments are classified as convergent, divergent, or transform.
- Terrain and hotspot features are currently generated as snapshots. Fracture and fusion rebuild the plate map rather than advance it through time.
- Current plate motion values and stress are not calibrated physical quantities and must not be treated as such without conversion.

## Simulation contract

### Units and time

- Planet radius is the distance unit for mesh positions. Store geographic positions on the unit sphere (or consistently divide/multiply by `SphereMesh.radius` at interfaces).
- Physical surface speed is expressed in **centimeters per year (cm/year)**. Base plate motion defaults to a seeded uniform range of 0–10 cm/year; the range is configurable.
- Simulation time is expressed in **millions of years (Myr)**.
- The initial fixed step is **1 Myr**. A step represents elapsed time; it is not a frame-rate-dependent update.
- Conversion: `1 cm/year × 1 Myr = 10 km`. The default Earth-like radius is 6,371 km. Store Euler angular velocity in radians per Myr; on the unit sphere, `|omega| = speed_cm_per_year × 10 / radius_km`.
- Use bounded substeps when a configured velocity would rotate a point too far in one update. The maximum angular movement per internal substep must be configurable and documented.

### Determinism and reset

- A world seed initializes all random generation used by the simulation, including plate motions, terrain, hotspot sources, and any later stochastic rules.
- Identical initial state, seed, parameters, and step count must produce identical simulation state.
- Reset restores the same generated initial world and simulation time `0`; it must not silently draw new random values.
- The simulation exposes elapsed time and completed step count.

### Persistent state

At minimum, simulation state must include:

- elapsed time and step count;
- plate identity and each plate's motion parameters;
- crust type (oceanic or continental), crust age, elevation, and crust thickness on the simulation grid;
- hotspot source locations in the mantle reference frame, distinct from volcano/terrain locations carried by plates;
- a seed and the parameter values needed to reproduce the run.

The state representation must preserve a complete spherical surface: every simulation cell has one owner and valid terrain/crust data after every step. It must not create holes, duplicate ownership, or redraw the plate map as a side effect of ordinary time advancement. Inverse remapping selects ownership from the nearest valid source face, interpolates continuous fields from up to four nearby source faces on that plate, and uses a deterministic nearest-candidate fallback where no valid source is nearby. Fallback cells are counted.

## Functional requirements

### M0 — Kinematic stepping (first implementation milestone)

1. Provide a simulation state/controller separate from rendering and procedural world generation.
2. Advance each rigid plate using its angular velocity and the fixed timestep. Rotate the plate's crust and terrain with it on the sphere.
3. Keep per-cell crust type, age, elevation, and thickness attached to the moving surface material. The initial implementation may use a documented resampling/remapping method, but it must preserve one owner per cell and report any remapping loss.
4. Recompute boundary geometry and relative motion from the updated plate state after each completed step. Boundary classification remains a label in M0; it does not yet deform terrain.
5. Keep hotspot sources fixed in the mantle frame. Existing volcanoes move with their plate. M0 may leave hotspot growth disabled, but it must not move the source with the crust.
6. Expose `step()`, `reset()`, `pause`, and `resume` operations. A single-step operation advances exactly one configured timestep regardless of rendering frame rate.
7. Provide a way to select simulation speed for continuous playback without changing the duration represented by one simulation step.
8. Show elapsed simulation time and playback state in the viewer.

### M1 — Boundary kinematics and crust creation/removal

- Derive convergence/divergence/transform motion from relative velocities projected onto the local boundary normal and tangent.
- Identify which side is subducting from the convergence direction and crust properties; do not infer polarity from speed alone.
- Add spreading at divergent oceanic boundaries, aging of oceanic crust, and removal of oceanic crust at subduction zones.
- Distinguish ocean-ocean, ocean-continent, and continent-continent convergence.
- Treat transform boundaries as horizontal shear/offset; do not create volcanic arcs solely from transform classification.

### M2 — Terrain response and feature lifecycle

- Apply bounded, tunable terrain processes: trench formation and arc uplift at subduction zones, crustal thickening and mountain uplift at continental collision zones, and rift/ridge relief at divergent zones.
- Grow hotspot volcanoes while over mantle sources; move formed material with its plate; then allow inactive volcanoes to subside and erode.
- Add mountain erosion and a simple isostatic response to crustal loading/unloading.
- Keep uplift, subsidence, erosion, and isostasy as separate parameters/processes so their effects can be inspected and calibrated independently.

## Non-functional requirements

- Simulation updates must not depend on render frame rate or wall-clock timing.
- Simulation parameters and units must be visible in code/API documentation and, where user-adjustable, in the viewer.
- Terrain changes must be bounded per step to avoid unstable elevation jumps.
- The chosen remapping approach must document its resolution limits and quantify conserved/lost quantities where practical (at least surface coverage and crustal material/age totals).
- Keep rendering colors and visual styling independent from simulation state.

## Prototype terrain and boundary behavior added after M0

- After each moving step, run one local majority pass over face ownership. A face changes plate only when the strongest neighboring label has a clear majority; its crust and terrain fields are copied from a neighboring face with that owner. This trims isolated zigzags and narrow notches without regenerating the whole map.
- Accumulate a narrow uplift belt at convergent boundaries. Continental collision has the strongest rate, ocean-continent convergence a lower rate, and ocean-ocean convergence the lowest rate. Spread the signal four face-neighbor rings inland with decay and cap relief at 0.45 mesh radii. These are deliberately visible prototype values, not calibrated geology.
- Treat the first point of each generated hotspot chain as a fixed mantle source. While its originating plate owns the source location, grow a mound there at 0.012 mesh radii per Myr, capped at 0.18 mesh radii and about 0.055 radians wide. Existing terrain is advected with the plate, so repeated eruptions can leave a chain behind as the plate moves.
- Hotspot growth pauses when another plate covers the source. Erosion, subsidence, subduction, and volcanic age progression remain future work.

## Acceptance criteria for M0

1. With zero angular velocity on every plate, stepping changes elapsed time and step count but does not change plate ownership, crust ages, or terrain.
2. A plate with known angular velocity moves a test point by the expected angular distance within the documented numerical tolerance.
3. An attached terrain feature remains at the same location relative to its plate after movement.
4. A mantle-fixed hotspot source remains stationary while a plate and its existing terrain move past it.
5. Replaying the same seeded initial world for the same number of steps yields identical state; reset followed by replay reproduces it.
6. Every step retains complete mesh coverage with one plate owner per cell; no cell becomes unassigned or multiply assigned.
7. One manual step advances exactly 1 Myr, independent of viewer frame rate; pause prevents automatic steps.

## Explicitly out of scope for M0

- Solving force balance, mantle flow, or self-consistent plate tectonics.
- Detailed fracture/fusion mechanics or creation of new plates at spreading ridges.
- Boundary-driven uplift, subduction, crust production/removal, erosion, or isostasy.
- Fine-scale non-rigid deformation within a plate. Rigid Euler rotation remains the baseline; additional local motion fields require a later model and separate validation.
- Calibration to a specific real planet. Earth-like values may be used as defaults, but remain adjustable assumptions.

## Open design decisions

1. **Material advection/remapping:** Use inverse mapping from destination cells to each plate's previous coordinates. The implementation must document coverage and resampling limits.
2. **Physical scale:** Use Earth-like radius 6,371 km by default, configurable for experiments.
3. **Initial motion distribution:** Randomize base plate speeds and directions from the world seed, with a configurable bounded speed range. Manual motion controls can be added later.
4. **Changing plate topology:** Keep plate identities fixed in M0; plate creation at divergence is deferred.
5. **UI scope:** Play, pause, single step forward/back, reset, and playback speed.

## Suggested implementation sequence

1. [x] Decide material advection and physical radius; formalize plate-motion units and seeded base motion initialization.
2. [x] Add a simulation state object with seeded initialization, fixed-step clock, reset, reverse stepping, playback controls, and rigid spherical rotation. Current material positions are stored per face; shared-mesh remapping remains step 3.
3. [x] Implement inverse material advection on the existing face grid, including ownership/coverage and area-weighted terrain, crust-volume, density-weighted crust-mass proxy, and crust-age diagnostics. Four-neighbor inverse-distance interpolation retains features smaller than a face, while remapping remains resolution-limited and may change integrals each step.
4. [x] Connect simulation state to viewer rendering; add play, pause, forward/back single-step, reset, playback-speed controls, and elapsed-time/playback status.
5. [x] Validate M0 behavior plus the requested boundary cleanup, convergent uplift, and mantle-fixed hotspot growth scenarios.
