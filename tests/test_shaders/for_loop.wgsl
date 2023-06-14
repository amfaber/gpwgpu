@group(0) @binding(0)
var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	#for IDK in 0..HI
		#for IDK2 in 0u..3u
	#IDK, #IDK2
		#endfor
	#endfor
}
