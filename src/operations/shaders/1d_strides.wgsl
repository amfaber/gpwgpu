// IDEAS
// Pass the current strides through push constants
// Dispatch completely flatly no matter what (n, 1, 1) and figure out accesses manually in the shader
// This allows a truely n-D kernel, which is specialized perfectly to the dimension at compile time
// As before, transpose the image on writing, allowing the first dim to always access with stride 1



struct PushConstants{
	// strides: array<u32, #N>,
	bounds: array<u32, #N>,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

var<push_constant> pc: PushConstants;

var<workgroup> local_storage: array<f32, #LOCALSIZE>;

@compute @workgroup_size(#WG_X, #WG_Y, #WG_Z)
fn main(
	@builtin(global_invocation_id) global_id: vec3<u32>,
	@builtin(local_invocation_index) local_index: u32,
) {
	if global_id.x >= arrayLength(&input){
		return;
	}
	
	var coords: array<i32, #N>;

	var strides: array<i32, #N>;

	var next_stride = 1;

	#for I in 0..N
	strides[#N - #I - 1] = next_stride;
	next_stride *= i32(pc.bounds[#I]);

	#endfor

	var idx = i32(global_id.x);

	#for I in 0..N
	coords[#I] = idx / strides[#I];
	idx = idx % strides[#I];

	#endfor

	idx = i32(globa_id.x);
	
	var acc = 0.0;
	
	#for I in -RINT..=RINT
	acc += input[idx + #I] * kernel_func();
	#endfor

}

