"""Battery degradation modelling for hybrid wind-battery power plants.

Modules
-------
xu               Xu et al. (2016) physical degradation model, used for
                 reporting the degradation actually incurred.
shi              Shi et al. (2018) convex polynomial surrogate, used where a
                 convex stress function is required.
subgradient      Exact per-timestep sub-gradient of the rainflow degradation
                 cost, and the frozen-dispatch capacity gradient.
site             Wind resource, turbine, and wake model setup.
economics        Discounting, capex, and revenue conventions shared by every
                 run script.
reporting        Console report of a single-year degradation result.
style            Shared matplotlib styling for thesis figures.
paths            Repository paths. Import from here rather than building
                 paths relative to a script.
"""

__version__ = "1.0.0"
