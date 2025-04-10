You primarily know about manim community edition (CE) v0.18 and below. We are using manim CE v0.19. Below are all relevant changes you need to know 
from this change. 

Manim v0.19.0 Code Generation Changes (from v0.18.x)
1. Breaking Changes (Code MUST Change):
Code Mobject:
Major rewrite (#4115). Interface is different. Cannot use old arguments/methods directly. Consult new Code documentation.
Use class method Code.get_styles_list() instead of styles_list attribute.
Method Parameter Renames:
ManimColor.from_hex(hex=...) -> ManimColor.from_hex(hex_str=...).
Scene.next_section(type=...) -> Scene.next_section(section_type=...).
Constructor Changes:
Sector: No longer takes inner_radius, outer_radius. Use radius. (AnnularSector still uses both).
Signature Changes:
SurroundingRectangle:
Accepts multiple Mobjects: SurroundingRectangle(mobj1, mobj2, ...).
Requires keyword arguments for options: SurroundingRectangle(mobj, color=RED, buff=0.1).
2. New Tools & Capabilities (Use These):
Color:
Use manim.HSV class for HSV colors.
New palettes available: DVIPSNAMES, SVGNAMES.
New ManimColor methods: .darker(), .lighter(), .contrasting().
Mobjects:
Initialize VGroup with generators/iterables: VGroup(Dot() for _ in range(5)).
New Mobjects available: ConvexHull, ConvexHull3D, Label, LabeledPolygram.
Animations:
Simulate typing with Write, AddTextLetterByLetter, RemoveTextLetterByLetter.
Set global defaults for animation types: Animation.set_default(run_time=2).
Use run_time=0 in animations (instant change).
Use Add(mobject) for instant appearance (like FadeIn(mobject, run_time=0)).
Use delay parameter in turn_animation_into_updater(animation, delay=...).
Coordinate Systems / Plotting:
Use colorscale argument in CoordinateSystem.plot().
Use @ shorthand with Axes: axes @ (x,y) (for c2p), axes @ point (for p2c).
Scene:
Access current animation time via self.time within a Scene.
VMobject:
Control stroke scaling: vmob.scale(2, scale_stroke=True).
LaTeX:
Expect better preservation of \color{} commands.

Make sure you're ALWAYS writing manim code under the assumption of using v0.19. Above are the changes relevant from v0.18 to v0.19. But 
also make sure you remember all changes relevant from older versions to v0.18. Make sure you don't use those even older paradigms as well.
Whenever you're about to use something from the above, recall that it's changed and REFERNCE THIS GUIDE to make sure we're not following old patterns.