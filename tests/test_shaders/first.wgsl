#import second

@group(0) @binding(0)
var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	let x = f32(#TEST);
	let idx = global_id.x + global_id.y * #N_COL;
	buffer[idx] = imported_function() * x;
}
