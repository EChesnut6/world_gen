This is going to be a small test program where I see how I can use rules to generate worlds.

This is going to be physics based, using a sphere with random (maybe not idk) tectonic plates and their directions and then using that info to create continents with additional randomness added in.
Things such as island chains, hot spots, or something else will be used to generate semi-realistic worlds.

I'm going to use python for it's familiarity and ease of using many valuable libraries, especially for drawing graphics.


Things to add:
- [ ] higher detail movement, the entire plate isn't moving at the exact same speed and direction in every part, especially if it's larger, spinning is also a possibilty. To implement this we need to add more vectors per plate dependent on the size. These vectors need to also include velocity, sometimes the plate is moving very slowly
- [ ] some of the colors aren't great yet
- [ ] new plate by divergence?
- [ ] instead of 3d axis, north pole and south pole indicators
- [ ] slight simulation of plate movement over time, nothing crazy complex needed here
- [ ] noise for plates to create a hight map. Can use this to determine where oceans are and where land is eventually giving us continents, from there we add in hotspots, island chains,  volano chains, and more for land creation