using GLVisualize, GLAbstraction, Reactive, GeometryTypes, Colors, GLWindow
import GLVisualize: slider, mm, button, labeled_slider, ScreenPartition, create_partitions!

window = glscreen()
partitions = ScreenPartition()
create_partitions!(partitions, window)
