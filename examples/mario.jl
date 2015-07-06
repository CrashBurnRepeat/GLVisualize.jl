using GeometryTypes, GLVisualize, GLAbstraction, GeometryTypes, ImageMagick, FileIO, ColorTypes, ImageIO, Reactive

type Mario{T}
    x 			::T
    y 			::T
    vx 			::T
    vy 			::T
    direction 	::Symbol
end

gravity(dt, mario) = (mario.vy = (mario.y > 0.0 ? mario.vy - (dt/4.0) : 0.0); mario)

function physics(dt, mario)
    mario.x = mario.x + dt * mario.vx
    mario.y	= max(0.0, mario.y + dt * mario.vy)
    mario
end

function walk(keys, mario)
    mario.vx = keys.x
    mario.direction = keys.x < 0.0 ? :left : keys.x > 0.0 ? :right : mario.direction
    mario
end

function jump(keys, mario)
    if keys.y > 0.0 && mario.vy == 0.0 
    	mario.vy = 6.0
    end
	mario
end

function update(dt, keys, mario)
    mario = gravity(dt, mario)
    mario = jump(keys, 	mario)
    mario = walk(keys, 	mario)
    mario = physics(dt, mario)
    mario
end



mario2model(mario) = translationmatrix(Vec3(mario.x, mario.y, 0f0))*scalematrix(Vec3(5f0))

const mario_images = Dict()
for verb in ["jump", "walk", "stand"], dir in ["left", "right"]
	mario_images[verb*dir] = read(File("imgs", "mario", verb, dir*".gif"))
end
function mario2image(mario, images=mario_images) 
	verb = mario.y > 0.0 ? "jump" : mario.vx != 0.0 ? "walk" : "stand"
	mario_images[verb*string(mario.direction)].value # is a signal of pictures itself (animation), so .value samples the current image
end
function arrows2vec(direction)
	direction == :up 	&& return Vector2( 0.0,  1.0)
	direction == :down 	&& return Vector2( 0.0, -1.0)
	direction == :right && return Vector2( 3.0,  0.0)
	direction == :left 	&& return Vector2(-3.0,  0.0)
	Vector2(0.0)
end

# Put everything together
arrows 			= sampleon(bounce(1:10), GLVisualize.ROOT_SCREEN.inputs[:arrow_navigation])
keys 			= lift(arrows2vec, arrows) 
mario_signal 	= lift(update, 8.0, keys, Mario(0.0, 0.0, 0.0, 0.0, :right))
image_stream 	= lift(mario2image, mario_signal)
modelmatrix 	= lift(mario2model, mario_signal)

view(visualize(image_stream, model=modelmatrix))
  
renderloop()