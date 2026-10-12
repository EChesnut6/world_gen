## Where the project is now

The code has useful ingredients for a terrain system, but it does not simulate plate movement over time yet. Plates have Euler rotation vectors, boundaries are classified from relative motion, and the viewer can trigger a fracture or fusion. Those operations rebuild the current plate map and terrain; there is no timestep that advances the world through millions of years.

Terrain is currently generated as a snapshot:

- Continental plates get a broad procedural landmass whose coverage is set by the land percentage control.
- Oceanic plates can get hotspot chains and island arcs. Their shapes and heights are generated once and saved.
- Ocean–ocean convergent boundaries can produce island arcs. Other boundary types do not yet build or deform terrain.
- Fracture and fusion alter plate identities, but they do not model long-term crust movement, uplift, subduction, or erosion.

The README still describes future work such as more detailed plate movement and a simple movement-over-time simulation. A useful next milestone is turning these static properties into a repeatable, measurable time step.

## Gaps that could make a simulation misleading

There are no time or distance units. Plate velocities are randomized values, and boundary classifications use fixed thresholds on those values. The code does not say how far a plate moves per step, or whether that step represents a year or a million years. Before evolving terrain, define units and a timestep. For scale, 1 cm/year corresponds to about 10 km per million years.

Plate boundaries are classified, but their effects are not simulated. The code labels convergence, divergence, and transform motion, but does not yet apply the corresponding deformation to the crust. Real convergent margins can produce trenches and volcanic arcs; continental collision can thicken crust and build large mountain belts. Transform motion mainly offsets and damages crust rather than creating a volcanic arc. USGS plate-tectonics overview (https://pubs.usgs.gov/gip/volc/tectonics.html), USGS plate-tectonics map (https://pubs.usgs.gov/pdf/planet.pdf).

The terrain is not attached to moving crust. A hotspot chain is saved in place today, and a fracture preserves its islands. In an advancing simulation, the land and seafloor must travel with their plate while hotspot sources stay in the mantle frame. Continued volcanism should form new seamounts over the source as older volcanoes move away, cool, subside, and erode. USGS hotspot overview (https://www.usgs.gov/faqs/what-a-hotspot-and-how-do-you-know-its-there), USGS Hawaiian volcano evolution (https://www.usgs.gov/observatories/hvo/evolution-hawaiian-volcanoes).

There is no crustal age or mass history. The model has crust thickness and density values, but not their spatial evolution. Oceanic crust should age and subside; new crust should form at spreading ridges; subduction should remove crust; and collisions can thicken continental crust. Without this history, terrain changes risk being arbitrary height edits.

Mountains have no lifecycle. There is no uplift rate, erosion, or isostatic adjustment. Mountains should rise under sustained compression, then be worn down and potentially rebound as material is removed. Hotspot volcanoes also grow, become inactive, subside, and erode. Hawaii’s older islands and seamounts record that progression, including subsidence under volcanic load. USGS growth and degradation of Hawaiian volcanoes (https://www.usgs.gov/publications/growth-and-degradation-hawaiian-volcanoes).

Current stress is a ranking signal, not physical stress. Boundary stress combines relative velocity and weakness with a heuristic. That is fine for choosing a fracture site, but should not directly become a force or uplift rate without calibration.

## Recommended roadmap

1. Define a simulation contract. Choose units, a fixed timestep such as 1 million years, deterministic random seeds, and state fields that persist: plate motion, crust type and age, elevation, and volcanic age. Decide whether the first version is kinematic—prescribed plate motions and terrain responses—or attempts to solve for forces. Kinematic is the practical starting point.
2. Add a kinematics-only time step. Advance each plate using its Euler rotation and the chosen timestep. Keep crust and terrain associated with the moving plate, track the time, and add pause, step, and reset controls. First verify that a hotspot track is carried along by its plate and that the simulated displacement matches its velocity and elapsed time.
3. Make boundary behavior depend on crust type and relative motion. Distinguish oceanic crust subducting beneath continental or oceanic crust from continent–continent collision. Add rifting at divergence and horizontal offsets at transform boundaries. Derive subduction polarity from the actual convergence direction, not a speed heuristic.
4. Add terrain response as separate, bounded processes. Start with uplift and subsidence fields around boundaries: trenches and volcanic arcs at subduction zones, broad thickening and mountain building at continental collisions, and rift valleys and ridges at divergence. Apply uplift from accumulated convergence, not from a one-time boundary label.
5. Give volcanoes and mountains lifecycles. Grow hotspots and boundary arcs over time; move their material with plates; then reduce elevation through erosion and subsidence. Add a simple isostatic response so adding or removing material changes crustal elevation. Keep these effects independently tunable.
6. Calibrate before adding complexity. Compare outputs against a few recognizable patterns: an age-progressive hotspot chain, an oceanic subduction arc, a continent–continent mountain belt, and a transform fault. Check conservation and stability across timestep sizes. This will reveal whether the model is producing consistent histories instead of attractive but disconnected terrain.

The main design decision to settle before coding is how crust moves across the sphere. The current mesh assigns each face to a plate, but a moving plate needs to carry its terrain and crustal age without leaving cracks, duplicating cells, or silently redrawing the plate map. Resolving that representation is the foundation for the rest of the simulation.