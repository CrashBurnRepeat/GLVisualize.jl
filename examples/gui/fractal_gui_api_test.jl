using GLVisualize, GLAbstraction, Reactive, GeometryTypes, Colors, GLWindow
import GLVisualize: slider, mm, button, labeled_slider
import GLVisualize: ScreenPartition
import GLVisualize: NullPartitionParams, AbsolutePartitionParams
import GLVisualize: PercentPartitionParams, TilePartitionParams
import GLVisualize: ControlPanel, WidgetParams, LabeledSliderParams, ButtonParams
import GLWindow: scaling_factor, nativewindow

w, h = (1000, 700)

window = glscreen(resolution = (w,h))

description = """
Demonstrating a UI for exploring the Koch snowflake.
"""
iconsize = 5mm
textsize = 5mm

custom_args1 = [(:color, RGBA{Float32}(0.0f0, 0.0f0, 0.0f0, 1f0)),
                (:stroke, (1f0, RGBA{Float32}(0.13f0, 0.13f0, 0.13f0, 1f0)))]
custom_args2 = [(:color, RGBA(0.0f0, 0.0f0, 0.0f0, 1f0))]

partitions = ScreenPartition(
    AbsolutePartitionParams(13*iconsize),
    NullPartitionParams(),
    window,
    [:sidebar,:main],
    options = [custom_args1, custom_args2]
)

pixel_scaling = scaling_factor(nativewindow(window))

sidebar_partitions = ScreenPartition(
    NullPartitionParams(),
    AbsolutePartitionParams(pixel_scaling[1]*h-13*iconsize),
    partitions.subscreens[:sidebar],
    [:controls, :mini],
    options = [custom_args1, custom_args2]
)

#Code for generating the controls
angles_range = 0.0:1.0:360.0
iterations_range = 1:11
colormap_range = map(RGBA{Float32}, colormap("Blues", 5))
thickness_value = Signal(0.4f0)
segments = Point2f0[
    (0.0, 0.0),
    (2 * iconsize, 0.0),
    (4 * iconsize, iconsize /  2),
    (6 * iconsize, iconsize / -2),
    (7 * iconsize, 0.0)
]
custom_slider_args = [
    (:text_scale, textsize),
    (:icon_size, iconsize),
    (:knob_scale, 3mm)
]
custom_colormap_args = [
    (:area, (12 * iconsize, iconsize/3)),
    (:knob_scale, 1.3mm)
]
custom_thickness_args = [
    (:text_scale, textsize),
    (:range, 0f0:0.05f0:20f0)
]
custom_segment_args = [
    (:knob_scale, 1.5mm)
]
custom_button_args = [
    (:relative_scale, iconsize)
]
names = [
    "angle 1",
    "angle 2",
    "angle 3",
    "angle 4",
    "iterations",
    "colormap",
    "thickness",
    "segment",
    "center cam"
]

controls = ControlPanel(
    [
    map(i->(LabeledSliderParams(angles_range, options = custom_slider_args)),1:4)...,
    LabeledSliderParams(iterations_range, options = custom_slider_args),
    WidgetParams(colormap_range, options = custom_colormap_args),
    WidgetParams(thickness_value, options = custom_thickness_args),
    WidgetParams(segments, options = custom_segment_args),
    ButtonParams("⛶", options = custom_button_args)
    ],
    sidebar_partitions.subscreens[:controls],
    names
)

# Add controls to subscreen
_view(visualize(
    controls.renderable,
    text_scale = textsize,
    width = 12iconsize
), sidebar_partitions.subscreens[:controls], camera = :fixed_pixel)

# Make local names for signals
angle_s = map(i->(controls.signals[i]), names[1:4])
iterations_s = controls.signals["iterations"]
cmap_s = controls.signals["colormap"]
thickness_s = controls.signals["thickness"]
line_s = controls.signals["segment"]
center_s = controls.signals["center cam"]

# Code for generating the fractal
function spin(dir, α, len)
    x, y = dir
    Point(x*cos(α) - y*sin(α), x*sin(α) + y*cos(α)) * len
end
function fractal_step!(
        a, b, depth, angles,
        result = Point2f0[a], levels = Float32[depth] # tmp too not allocate
    )
    depth == 0 && return result, levels, b
    N = length(angles)
    diff = (b - a)
    len = norm(diff) / (N - 1)
    nth_segment = normalize(diff)
    for n = 1:N
        b = a + spin(nth_segment * len, angles[n]...) # rotate with current angle
        _, _, b = fractal_step!(a, b, depth-1, angles, result, levels) # recursion step
        if n < N
            push!(result, b)
            push!(levels, depth)
        end
        nth_segment = normalize(b - a)
        a = b
    end
    result, levels, b
end

const T = Float64
const P = Point{2, T}

function generate_fractal(angles, depth = 5)
    tmp = zeros(P, length(angles))
    angles = map(x-> (deg2rad(T(x[1])), T(x[2])), angles)
    result, levels, b = fractal_step!(P(0,0), P(300,0), round(Int, depth), angles)
    push!(result, b)
    push!(levels, depth)
    mini, maxi = extrema(result)
    w = 1 ./ maximum(maxi - mini)
    map!(result, result) do p
        1000 * (p - mini) .* w
    end
    # invert
    result, levels
end

function to_anglelengths(angles, line)
    diff0 = Point2f0(1, 0)
    v1 = first(line)
    maxlen = 0
    for (i, v2) in enumerate(line[2:end])
        diff1 = v2 - v1
        len = norm(diff1)
        diff1 = normalize(diff1)
        maxlen = max(len, maxlen)

        d = dot(diff0, diff1)
        det = cross(diff0, diff1)

        angles[i] = (atan2(det, d), len)
        v1 = v2; diff0 = diff1
    end
    for (i, al) in enumerate(angles)
        angles[i] = (rad2deg(al[1]), al[2] / maxlen)
    end
    angles
end
v0 = to_anglelengths(Array{Tuple{Float32, Float32}}(4), value(line_s))

angle_vec1 = foldp(to_anglelengths, v0, line_s)

anglevec2 = foldp(Array{Tuple{Float32, Float32}}(4), angle_s...) do angles, s...
    for i=1:4
        angles[i] = s[i], 1.0
    end
    angles
end
anglevec = merge(angle_vec1, anglevec2)

it1_points = map(anglevec) do angles
    generate_fractal(angles, 1)[1] ./ 5f0
end

line_level = map(anglevec, iterations_s) do angles, iter
    generate_fractal(angles, iter)
end
line_pos = map(first, line_level)

# Add fractals to subscreen
_view(visualize(
    line_pos, :lines,
    thickness = thickness_s,
    color_map = cmap_s,
    color_norm = map(i-> Vec2f0(0, i), iterations_s),
    intensity = map(last, line_level),
    boundingbox = nothing
), partitions.subscreens[:main], camera = :orthographic_pixel)

_view(visualize(
    it1_points,
    :lines,
    model = translationmatrix(Vec3f0(13*iconsize/4, 13*iconsize/4, 0)),
    thickness = 2f0,
    color = RGBA(1f0, 1f0, 1f0, 1f0)
), sidebar_partitions.subscreens[:mini], camera = :fixed_pixel)

# Signal housekeeping and render
const cam = partitions.subscreens[:main].cameras[:orthographic_pixel]

s = preserve(map(center_s) do clicked
    clicked && center!(cam, AABB(value(line_pos)))
    nothing
end)

# center won't get executed before the first time the center button is clicked.
# we still want to start centered ;)
center!(cam, AABB(value(line_pos)))

if !isdefined(:runtests)
    renderloop(window)
    # clean up signals
    map(i->close(controls.signals[i], false),names[1:4])
    close(controls.signals["center cam"], false)
    close(controls.signals["thickness"], false)
    close(controls.signals["iterations"], false)
end
