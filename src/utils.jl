"""
Determines if of Image Type
"""
function isa_image(x::Type{T}) where T<:Matrix
    eltype(T) <: Union{Colorant, Colors.Fractional}
end
isa_image(x::Matrix) = isa_image(typeof(x))
# if !isdefined(Images, :ImageAxes)
#     include_string("""
#     isa_image(x::Images.Image) = true
#     """)
# end
isa_image(x) = false

# Splits a dictionary in two dicts, via a condition
function Base.split(condition::Function, associative::Associative)
    A = similar(associative)
    B = similar(associative)
    for (key, value) in associative
        if condition(key, value)
            A[key] = value
        else
            B[key] = value
        end
    end
    A, B
end


function assemble_robj(data, program, bb, primitive, pre_fun, post_fun)
    pre = if pre_fun != nothing
        () -> (GLAbstraction.StandardPrerender(); pre_fun())
    else
        GLAbstraction.StandardPrerender()
    end
    robj = RenderObject(data, program, pre, nothing, bb, nothing)
    post = if haskey(data, :instances)
        GLAbstraction.StandardPostrenderInstanced(data[:instances], robj.vertexarray, primitive)
    else
        GLAbstraction.StandardPostrender(robj.vertexarray, primitive)
    end
    robj.postrenderfunction = if post_fun != nothing
        () -> begin
            post()
            post_fun()
        end
    else
        post
    end
    robj
end


function assemble_shader(data)
    shader = data[:shader]
    delete!(data, :shader)
    default_bb = Signal(centered(AABB))
    bb  = get(data, :boundingbox, default_bb)
    if bb == nothing || isa(bb, Signal{Void})
        bb = default_bb
    end
    glp = get(data, :gl_primitive, GL_TRIANGLES)
    robj = assemble_robj(
        data, shader, bb, glp,
        get(data, :prerender, nothing),
        get(data, :postrender, nothing)
    )
    Context(robj)
end




function y_partition_abs(area, amount::T) where T
    a = round(Int, amount)
    x = area.value.x
    y = area.value.y
    p = const_lift(area) do r
        (
            SimpleRectangle{Int}(x, y,     r.w, a),
            SimpleRectangle{Int}(x, y + a, r.w, r.h - a)
        )
    end
    return map(first, p), map(last, p)
end
function y_partition_abs(area, amount::Array{T}) where T
    partitions = Array{Signal{SimpleRectangle{Int}}}(length(amount)+1)
    temp_area = area
    adjust = 0
    for x in 1:length(amount)
        partitions[x], second_area = y_partition_abs(temp_area, amount[x]-adjust)
        adjust = amount[x]
        temp_area = second_area
    end
    partitions[end]=temp_area
    return Tuple(partitions)
end
function x_partition_abs(area, amount::T) where T<:Real
    a = round(Int, amount)
    x = area.value.x
    y = area.value.y
    p = const_lift(area) do r
        (
            SimpleRectangle{Int}(x,     y, a,       r.h),
            SimpleRectangle{Int}(x + a, y, r.w - a, r.h)
        )
    end
    return map(first, p), map(last, p)
end
function x_partition_abs(area, amount::Array{T}) where T
    partitions = Array{Signal{SimpleRectangle{Int}}}(length(amount)+1)
    temp_area = area
    adjust = 0
    for x in 1:length(amount)
        partitions[x], second_area = x_partition_abs(temp_area, amount[x]-adjust)
        adjust = amount[x]
        temp_area = second_area
    end
    partitions[end]=temp_area
    return Tuple(partitions)
end
function y_partition(area, percent::T) where T<:Real
    amount = percent / 100.0
    x = area.value.x
    y = area.value.y
    p = const_lift(area) do r
        (
            SimpleRectangle{Int}(x, y,                          r.w, round(Int, r.h*amount)),
            SimpleRectangle{Int}(x, y + round(Int, r.h*amount), r.w, round(Int, r.h*(1-amount)))
        )
    end
    return map(first, p), map(last, p)
end
function y_partition(area, percent::Array{T}) where T<:Real
    amount = percent / 100.0
    partitions = Array{Signal{SimpleRectangle{Int}}}(length(amount)+1)
    temp_area = area
    adjust = 1.0
    for x in 1:length(amount)
        partitions[x], second_area = y_partition_abs(temp_area, amount[x]/adjust)
        adjust = 1.0-amount[x]
        temp_area = second_area
    end
    partitions[end]=temp_area
    return Tuple(partitions)
end
function x_partition(area, percent::T) where T<:Real
    amount = percent / 100.0
    x = area.value.x
    y = area.value.y
    p = const_lift(area) do r
        (
            SimpleRectangle{Int}(x,                          y, round(Int, r.w*amount), r.h ),
            SimpleRectangle{Int}(x + round(Int, r.w*amount), y, round(Int, r.w*(1-amount)), r.h)
        )
    end
    return map(first, p), map(last, p)
end
function x_partition(area, percent::Array{T}) where T<:Real
    amount = percent / 100.0
    partitions = Array{Signal{SimpleRectangle{Int}}}(length(amount)+1)
    temp_area = area
    adjust = 1.0
    for x in 1:length(amount)
        partitions[x], second_area = x_partition_abs(temp_area, amount[x]/adjust)
        adjust = 1.0-amount[x]
        temp_area = second_area
    end
    partitions[end]=temp_area
    return Tuple(partitions)
end

function x_partition_tile(area, count)
    tiles = Array{Signal{SimpleRectangle{Int}}}(count)
    temp_area = area
    for x in count:-1:2
        percent = 100.0/x
        tiles[end-x+1], second_area = x_partition(temp_area, percent)
        temp_area = second_area
    end
    tiles[end] = temp_area
    return Tuple(tiles)
end

function y_partition_tile(area, count)
    tiles = Array{Signal{SimpleRectangle{Int}}}(count)
    temp_area = area
    for x in count:-1:2
        percent = 100.0/x
        tiles[end-x+1], second_area = y_partition(temp_area, percent)
        temp_area = second_area
    end
    tiles[end] = temp_area
    return Tuple(tiles)
end

create_x_partition_shapes(area, params::PercentPartitionParams) =
    x_partition(area,params.value)

create_x_partition_shapes(area, params::AbsolutePartitionParams) =
    x_partition_abs(area,params.value)

create_x_partition_shapes(area, params::TilePartitionParams) =
    x_partition_tile(area,param.value)

create_y_partition_shapes(area, params::PercentPartitionParams) =
    y_partition(area,params.value)

create_y_partition_shapes(area, params::AbsolutePartitionParams) =
    y_partition_abs(area,params.value)

create_y_partition_shapes(area, params::TilePartitionParams) =
    y_partition_tile(area,param.value)

function create_partition_shapes(area, x_params, y_params)
    if !isa(x_params, NullPartitionParams)
        x_partitions = create_x_partition_shapes(area,x_params)
    else
        x_partitions = Tuple([area]) #ensures that y logic works
    end
    if !isa(y_params, NullPartitionParams)
        if isa(y_params, TilePartitionParams)
            num_y_partitions = y_params.value
        else
            num_y_partitions = length(y_params.value)+1
        end
        y_partitions = Array{Any}(num_y_partitions)
        for x in 1:length(x_partitions)
            y_partitions[x] = create_y_partition_shapes(x_partitions[x],y_params)
        end
        return Tuple(y_partitions)
    else
        return x_partitions
    end
end


function create_partitions(
    x_partition_params::P1,
    y_partition_params::P2,
    window=current_screen(),
    names=[],
    ;options=[]
    ) where {P1<:PartitionParams, P2<:PartitionParams}
    partition_shapes = create_partition_shapes(
        window.area,
        x_partition_params,
        y_partition_params
    )
    subscreens = Dict()
    shape_idx = 1
    x_length = length(partition_shapes)
    y_length = isa(partition_shapes[1], Tuple) ?
        length(partition_shapes[1]) : 1
    num_shapes = x_length*y_length #assumes grid
    if num_shapes>length(names)
        names_temp = collect(1:num_shapes)
        names_temp[1:length(names)] = names
        names = names_temp
    end
    num_options = length(options)
    if num_shapes>num_options
        options_temp = fill([],size(options))
        options_temp[1:length(options)]=screen_options
        options = options_temp
    end
    for x_partition in partition_shapes
        if isa(x_partition, Tuple)
            for y_partition in x_partition
                subscreens[names[shape_idx]] =
                    Screen(window, area=y_partition; options[shape_idx]...)
                shape_idx+=1
            end
        else
            subscreens[names[shape_idx]] =
                Screen(window, area=x_partition; options[shape_idx]...)
            shape_idx+=1
        end
    end
    return subscreens
end

function create_controls(control_params, window)
    if isa(control_params, Array)
        control_objs = ntuple(length(control_params)) do i
            widget_dispatch(control_params[i],window)
        end
    else
        (control_objs,) = widget_dispatch(control_params,window)
    end
    return control_objs
end

widget_dispatch(control_param::WidgetParams, window) =
    widget(control_param.value, window; control_param.options...)
widget_dispatch(control_param::LabeledSliderParams, window) =
    labeled_slider(control_param.value, window; control_param.options...)
widget_dispatch(control_param::ButtonParams, window) =
    button(control_param.value, window; control_param.options...)

function create_control_signals(control_objs, names)
    signals = Dict()
    for i in 1:length(control_objs)
        signals[names[i]] = control_objs[i][2]
    end
    return signals
end

function create_control_renderable(control_objs,names)
    renderable=Array{Pair}(length(control_objs))
    for i in 1:length(renderable)
        renderable[i] = names[i]=>control_objs[i][1]
    end
    return renderable
end

glboundingbox(mini, maxi) = AABB{Float32}(Vec3f0(mini), Vec3f0(maxi)-Vec3f0(mini))
function default_boundingbox(main, model)
    main == nothing && return Signal(AABB{Float32}(Vec3f0(0), Vec3f0(1)))
    const_lift(*, model, AABB{Float32}(main))
end
AABB(a::GPUArray) = AABB{Float32}(gpu_data(a))
AABB{T}(a::GPUArray) where {T} = AABB{T}(gpu_data(a))


"""
Returns two signals, one boolean signal if clicked over `robj` and another
one that consists of the object clicked on and another argument indicating that it's the first click
"""
function clicked(robj::RenderObject, button::MouseButton, window::Screen)
    @materialize mouse_hover, mouse_buttons_pressed = window.inputs
    leftclicked = const_lift(mouse_hover, mouse_buttons_pressed) do mh, mbp
        mh[1] == robj.id && mbp == Int[button]
    end
    clicked_on_obj = keepwhen(leftclicked, false, leftclicked)
    clicked_on_obj = const_lift((mh, x)->(x,robj,mh), mouse_hover, leftclicked)
    leftclicked, clicked_on_obj
end

"""
Returns a boolean signal indicating if the mouse hovers over `robj`
"""
is_hovering(robj::RenderObject, window::Screen) =
    droprepeats(const_lift(is_same_id, mouse2id(window), robj))

function dragon_tmp(past, mh, mbp, mpos, robj, button, start_value)
    diff, dragstart_index, was_clicked, dragstart_pos = past
    over_obj = mh[1] == robj.id
    is_clicked = mbp == Int[button]
    if is_clicked && was_clicked # is draggin'
        return (dragstart_pos-mpos, dragstart_index, true, dragstart_pos)
    elseif over_obj && is_clicked && !was_clicked # drag started
        return (Vec2f0(0), mh[2], true, mpos)
    end
    return start_value
end

"""
Returns a signal with the difference from dragstart and current mouse position,
and the index from the current ROBJ id.
"""
function dragged_on(robj::RenderObject, button::MouseButton, window::Screen)
    @materialize mouse_buttons_pressed, mouseposition = window.inputs
    mousehover = mouse2id(window)
    mousedown = const_lift(GLAbstraction.singlepressed, mouse_buttons_pressed, GLFW.MOUSE_BUTTON_LEFT)
    condition = const_lift(is_same_id, mousehover, robj)
    dragg = GLAbstraction.dragged(mouseposition, mousedown, condition)
    filterwhen(mousedown, (value(dragg), 0), map(dragg) do d
        d, value(mousehover).index
    end)
end

points2f0(positions::Vector{T}, range::Range) where {T} = Point2f0[Point2f0(range[i], positions[i]) for i=1:length(range)]

extrema2f0(x::Array{T,N}) where {T<:Intensity,N} = Vec2f0(extrema(reinterpret(Float32,x)))
extrema2f0(x::Array{T,N}) where {T,N} = Vec2f0(extrema(x))
extrema2f0(x::GPUArray) = extrema2f0(gpu_data(x))
function extrema2f0(x::Array{T,N}) where {T<:Vec,N}
    _norm = map(norm, x)
    Vec2f0(minimum(_norm), maximum(_norm))
end

function mix_linearly(a::C, b::C, s) where C<:Colorant
    RGBA{Float32}((1-s)*comp1(a)+s*comp1(b), (1-s)*comp2(a)+s*comp2(b), (1-s)*comp3(a)+s*comp3(b), (1-s)*alpha(a)+s*alpha(b))
end

color_lookup(cmap, value, mi, ma) = color_lookup(cmap, value, (mi, ma))
function color_lookup(cmap, value, color_norm)
    mi,ma = color_norm
    scaled = clamp((value-mi)/(ma-mi), 0, 1)
    index = scaled * (length(cmap)-1)
    i_a, i_b = floor(Int, index)+1, ceil(Int, index)+1
    mix_linearly(cmap[i_a], cmap[i_b], scaled)
end


"""
Converts index arrays to the OpenGL equivalent.
"""
to_index_buffer(x::GLBuffer) = x
to_index_buffer(x::TOrSignal{Int}) = x
to_index_buffer(x::VecOrSignal{UnitRange{Int}}) = x
to_index_buffer(x::TOrSignal{UnitRange{Int}}) = x
"""
For integers, we transform it to 0 based indices
"""
to_index_buffer(x::Vector{I}) where {I<:Integer} = indexbuffer(map(i-> Cuint(i-1), x))
function to_index_buffer(x::Signal{Vector{I}}) where I<:Integer
    x = map(x-> Cuint[i-1 for i=x], x)
    gpu_mem = GLBuffer(value(x), buffertype = GL_ELEMENT_ARRAY_BUFFER)
    preserve(const_lift(update!, gpu_mem, x))
    gpu_mem
end
"""
If already GLuint, we assume its 0 based (bad heuristic, should better be solved with some Index type)
"""
to_index_buffer(x::Vector{I}) where {I<:GLuint} = indexbuffer(x)
function to_index_buffer(x::Signal{Vector{I}}) where I<:GLuint
    gpu_mem = GLBuffer(value(x), buffertype = GL_ELEMENT_ARRAY_BUFFER)
    preserve(const_lift(update!, gpu_mem, x))
    gpu_mem
end
function to_index_buffer(x::Signal{Vector{I}}) where I <: Face{2, GLIndex}
    gpu_mem = GLBuffer(value(x), buffertype = GL_ELEMENT_ARRAY_BUFFER)
    preserve(const_lift(update!, gpu_mem, x))
    gpu_mem
end
to_index_buffer(x) = error(
    "Not a valid index type: $(typeof(x)).
    Please choose from Int, Vector{UnitRange{Int}}, Vector{Int} or a signal of either of them"
)

"""
Creates a moving average and discards values to close together.
If discarded return (false, p), if smoothed, (true, smoothed_p).
"""
function moving_average(p, cutoff,  history, n = 5)
    if length(history) > 0
        if norm(p - history[end]) < cutoff
            return false, p # don't keep point
        end
    end
    if length(history) == 5
        # maybe better to just keep a current index
        history[1:5] = circshift(view(history, 1:5), -1)
        history[end] = p
    else
        push!(history, p)
    end
    true, sum(history) ./ length(history)# smooth
end

function layoutlinspace(n::Integer)
    if n == 1
        1:1
    else
        linspace(1/n, 1, n)
    end
end
xlayout(x::Int) = zip(layoutlinspace(x), Iterators.repeated(""))
function xlayout(x::AbstractVector{T}) where T <: AbstractFloat
    zip(x, Iterators.repeated(""))
end

function xlayout(x::AbstractVector)
    zip(layoutlinspace(length(x)), x)
end
function ylayout(x::AbstractVector)
    zip(layoutlinspace(length(x)), x)
end
function ylayout(x::AbstractVector{T}) where T <: Tuple
    sizes = map(first, x)
    values = map(last, x)
    zip(sizes, values)
end
function IRect(x, y , w, h)
    SimpleRectangle(
        round(Int, x),
        round(Int, y),
        round(Int, w),
        round(Int, h),
    )
end

function layout_rect(area, lastw, lasth, w, h)
    wp = widths(area)
    xmin = wp[1] * lastw
    ymin = wp[2] * lasth
    xmax = wp[1] * w
    ymax = wp[2] * h
    xmax = max(xmin, xmax)
    xmin = min(xmin, xmax)
    ymax = max(ymin, ymax)
    ymin = min(ymin, ymax)
    IRect(xmin, ymin, xmax - xmin, ymax - ymin)
end

function layoutscreens(parent, layout;
        title_height = 6mm, text_color = default(RGBA),
        background_color = RGBA(1f0, 1f0, 1f0), stroke_color = RGBA(0f0, 0f0, 0f0, 0.4f0),
        kw_args...
    )
    layout = reverse(layout) # we start from bottom to top, while lists are written top to bottom
    lastw, lasth = 0, 0
    result = Vector{Screen}[]
    for (h, xlist) in ylayout(layout)
        result_x = Screen[]
        for (w, title) in xlayout(xlist)
            area = const_lift(layout_rect, parent.area, lastw, lasth, w, h)

            screen = Screen(parent; area = area, kw_args...)
            push!(result_x, screen)
            if !isempty(value(title))
                tarea = map(area) do area
                    IRect(0, area.h - title_height, area.w, title_height)
                end
                title_screen = Screen(screen, area = tarea, color = background_color)
                robj = visualize(
                    title, relative_scale = title_height * 0.7,
                    direction = 1, gap = Vec3f0(1mm, 0, 0),# in case it's a list!
                    color = text_color
                )
                gap = title_height * 0.15
                GLAbstraction.transform!(robj, translationmatrix(Vec3f0(gap, gap, 0)))
                _view(robj, title_screen, camera = :fixed_pixel)
                if stroke_color != nothing
                    _view(visualize(
                        map(a-> Point2f0[(0, 0), (a.w, 0)], area), :linesegment,
                        thickness = 1f0, color = stroke_color
                    ), title_screen, camera = :fixed_pixel)
                end
            end
            lastw = w
        end
        lastw = 0; lasth = h
        push!(result, result_x)
    end
    result
end
